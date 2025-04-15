"""Equivariant transformer for multivector data."""

import time
from dataclasses import replace
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.models.models import Model
from ..layers.attention.config import SelfAttentionConfig
from ..layers.lgatr_block import LGATrBlock
from ..layers.linear import EquiLinear
from ..layers.mlp.config import MLPConfig
from ..interface import (
    embed_vector,
    extract_scalar,
)

TYPE_TOKEN_DICT = {
    "gg_4g": [0, 1, 2, 3, 4, 5],
    "gg_5g": [0, 1, 2, 3, 4, 5, 6],
    "gg_6g": [0, 1, 2, 3, 4, 5, 6, 7],
    "gg_7g": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "gg_qqbar2g": [0, 0, 1, 2, 3, 3],
    "gg_qqbar3g": [0, 0, 1, 2, 3, 3, 3],
    "gg_qqbar4g": [0, 0, 1, 2, 3, 3, 3, 3],
    "gg_qqbar5g": [0, 0, 1, 2, 3, 3, 3, 3, 3],
}


def encode_tokens(type_token, global_token, token_size, isgatr, batchsize, device):
    """Compute embedded type_token and global_token to be used within Transformers

    Parameters
    type_token: iterable of int
        list with type_tokens for each particle in the event
    global_token: int
    isgatr: bool
        whether the encoded tokens will be used within L-GATr or within the baseline Transformer
        This affects how many zeroes have to be padded to the global_token (4 more for the baseline Transformer)
    batchsize: int
    device: torch.device


    Returns:
    type_token: torch.Tensor with shape (batchsize, num_particles, type_token_max)
        one-hot-encoded type tokens, to be appended to each encoded 4-momenta in case of the
        baseline transformer / make up the full scalar channel for L-GATr
    global_token: torch.Tensor with shape (batchsize, 1, type_token_max+4)
        ont-hot-encoded dataset token, this will be the global_token and appended to the individual particles
    """
    type_token = nn.functional.one_hot(type_token, num_classes=token_size)
    type_token = type_token.expand(batchsize, *type_token.shape[1:]).float()

    global_token = nn.functional.one_hot(
        global_token, num_classes=token_size + (0 if isgatr else 4)
    )
    global_token = global_token.expand(batchsize, *global_token.shape[1:]).float()
    return type_token, global_token


class LGATr(Model):
    """
    Wrapper that handles interface to the GATr code
    - create dataclasses for attention and mlp
    - append spurions (symmetry-breaking)
    - interface to geometric algebra
    - extract tagging score with global token or mean-aggregation
    """

    def __init__(self, logger, process, cfg, dims_in, dims_out, model_path, device):
        super().__init__(logger, cfg, dims_in, dims_out, model_path, device)

        in_mv_channels = cfg.model["in_mv_channels"]
        out_mv_channels = cfg.model["out_mv_channels"]
        hidden_mv_channels = cfg.model["hidden_mv_channels"]
        in_s_channels = cfg.model.get("in_s_channels", None)
        out_s_channels = cfg.model.get("out_s_channels", None)
        hidden_s_channels = cfg.model.get("hidden_s_channels", None)
        attention = cfg.model["attention"]
        mlp = cfg.model["mlp"]
        num_blocks = cfg.model.get("num_blocks", 10)
        global_token = cfg.model.get("global_token", True)
        dropout_prob = cfg.model.get("dropout_prob", None)
        double_layernorm = cfg.model.get("double_layernorm", False)
        checkpoint_blocks = cfg.model.get("checkpoint_blocks", False)

        self.type_token = TYPE_TOKEN_DICT[process]
        token_size = max(self.type_token) + 1
        permutation_invariance = cfg.model.get(
            "permutation_invariance", True
        )  # not used for now
        in_s_channels = token_size
        self.global_token = global_token
        self.loss_fct = (
            nn.L1Loss() if cfg.train.get("loss", "MSE") == "L1" else nn.MSELoss()
        )

        self.net = LGATr_net(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            attention=attention,
            mlp=mlp,
            num_blocks=num_blocks,
            checkpoint_blocks=checkpoint_blocks,
            dropout_prob=dropout_prob,
            double_layernorm=double_layernorm,
        )
        self.amplitude_wrapper = AmplitudeWrapper(self.net, token_size)

    def forward(self, x):
        return self.amplitude_wrapper.forward(
            x,
            type_token=torch.tensor([self.type_token for _ in range(x.shape[0])])
            .to(torch.int64)
            .to(x.device),
            global_token=torch.zeros(x.shape[0], 1).to(torch.int64).to(x.device),
        ).squeeze(0)

    def predict(self, x):
        return self.forward(x)


class LGATr_net(nn.Module):
    """L-GATr network for a data with a single token dimension.

    It combines `num_blocks` L-GATr transformer blocks, each consisting of geometric self-attention
    layers, a geometric MLP, residual connections, and normalization layers. In addition, there
    are initial and final equivariant linear layers.

    Assumes input has shape `(..., items, in_channels, 16)`, output has shape
    `(..., items, out_channels, 16)`, will create hidden representations with shape
    `(..., items, hidden_channels, 16)`.

    Parameters
    ----------
    in_mv_channels : int
        Number of input multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    hidden_mv_channels : int
        Number of hidden multivector channels.
    in_s_channels : None or int
        If not None, sets the number of scalar input channels.
    out_s_channels : None or int
        If not None, sets the number of scalar output channels.
    hidden_s_channels : None or int
        If not None, sets the number of scalar hidden channels.
    attention: Dict
        Data for SelfAttentionConfig
    mlp: Dict
        Data for MLPConfig
    num_blocks : int
        Number of transformer blocks.
    dropout_prob : float or None
        Dropout probability
    double_layernorm : bool
        Whether to use double layer normalization
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: Optional[int],
        out_s_channels: Optional[int],
        hidden_s_channels: Optional[int],
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        num_blocks: int = 10,
        reinsert_mv_channels: Optional[Tuple[int]] = None,
        reinsert_s_channels: Optional[Tuple[int]] = None,
        checkpoint_blocks: bool = False,
        dropout_prob: Optional[float] = None,
        double_layernorm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )
        attention = replace(
            SelfAttentionConfig.cast(attention),
            additional_qk_mv_channels=0
            if reinsert_mv_channels is None
            else len(reinsert_mv_channels),
            additional_qk_s_channels=0
            if reinsert_s_channels is None
            else len(reinsert_s_channels),
        )
        mlp = MLPConfig.cast(mlp)
        self.blocks = nn.ModuleList(
            [
                LGATrBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    attention=attention,
                    mlp=mlp,
                    dropout_prob=dropout_prob,
                    double_layernorm=double_layernorm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=out_s_channels,
        )
        self._reinsert_s_channels = reinsert_s_channels
        self._reinsert_mv_channels = reinsert_mv_channels
        self._checkpoint_blocks = checkpoint_blocks

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
        **attn_kwargs,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass of the network.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors.
        scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars.
        **attn_kwargs
            Optional keyword arguments passed to attention.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors.
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """

        # Channels that will be re-inserted in any query / key computation
        (
            additional_qk_features_mv,
            additional_qk_features_s,
        ) = self._construct_reinserted_channels(multivectors, scalars)

        # Pass through the blocks
        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)
        for block in self.blocks:
            if self._checkpoint_blocks:
                h_mv, h_s = checkpoint(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    **attn_kwargs,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    **attn_kwargs,
                )

        outputs_mv, outputs_s = self.linear_out(h_mv, scalars=h_s)

        return outputs_mv, outputs_s

    def _construct_reinserted_channels(self, multivectors, scalars):
        """Constructs input features that will be reinserted in every attention layer."""

        if self._reinsert_mv_channels is None:
            additional_qk_features_mv = None
        else:
            additional_qk_features_mv = multivectors[..., self._reinsert_mv_channels, :]

        if self._reinsert_s_channels is None:
            additional_qk_features_s = None
        else:
            assert scalars is not None
            additional_qk_features_s = scalars[..., self._reinsert_s_channels]

        return additional_qk_features_mv, additional_qk_features_s


class AmplitudeWrapper(nn.Module):
    def __init__(self, net, token_size, reinsert_type_token=False):
        super().__init__()
        self.net = net
        self.token_size = token_size

    def forward(self, inputs: torch.Tensor, type_token, global_token, attn_mask=None):
        multivector, scalars = self.embed_into_ga(inputs, type_token, global_token)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attn_mask=attn_mask
        )
        amplitude = self.extract_from_ga(multivector_outputs, scalar_outputs)
        return amplitude

    def embed_into_ga(self, inputs, type_token, global_token):
        batchsize, _ = inputs.shape
        num_objects = _ // 4
        inputs = inputs.unsqueeze(0)
        inputs = inputs.view(1, batchsize, num_objects, 4)
        nprocesses, batchsize, num_objects, _ = inputs.shape

        # encode momenta in multivectors
        multivector = embed_vector(inputs)
        multivector = multivector.unsqueeze(-2)

        type_token, global_token = encode_tokens(
            type_token,
            global_token,
            self.token_size,
            isgatr=True,
            batchsize=batchsize,
            device=inputs.device,
        )
        type_token = type_token.to(inputs.dtype)
        global_token = global_token.to(inputs.dtype)

        # encode type_token in scalars
        scalars = type_token

        # global token
        global_token_mv = torch.zeros(
            (nprocesses, batchsize, 1, multivector.shape[-2], multivector.shape[-1]),
            dtype=multivector.dtype,
            device=multivector.device,
        )
        global_token_s = global_token
        multivector = torch.cat((global_token_mv, multivector), dim=-3)
        scalars = torch.cat((global_token_s, scalars), dim=-2)
        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        # Extract scalars from GA representation
        lorentz_scalars = extract_scalar(multivector)[..., 0]

        amplitude = lorentz_scalars[..., 0, :]
        return amplitude
