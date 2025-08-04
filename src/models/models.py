import torch
from torch import nn
from .train import Model


class MLP(Model):
    def __init__(
        self,
        logger,
        process,
        cfg,
        dims_in,
        helicity_dict_size,
        dims_out,
        model_path,
        device,
    ):
        super().__init__(logger, cfg, dims_in, dims_out, model_path, device)
        self.helicity_dict_size = helicity_dict_size
        self.init_loss()
        self.init_net()

    def init_net(self):
        if self.cfg.model.get("activation", "relu") == "relu":
            activation = nn.ReLU()
        elif self.cfg.model.get("activation", "relu") == "gelu":
            activation = nn.GELU()
        elif self.cfg.model.get("activation", "relu") == "tanh":
            activation = nn.Tanh()
        else:
            raise NotImplementedError(
                f"Activation function {self.cfg.model.get('activation', 'relu')} not implemented"
            )

        if self.cfg.dataset.embed_helicities.get("use", False):
            feature_layers = []
            feature_layers.append(
                nn.Linear(self.dims_in - 1, self.cfg.model["internal_size"])
            )
            feature_layers.append(activation)
            feature_embed = nn.Sequential(*feature_layers)

            hel_layers = []
            hel_layers.append(
                nn.Embedding(
                    self.helicity_dict_size,
                    self.cfg.dataset.embed_helicities.get("embed_dimension", 64),
                )
            )
            hel_layers.append(activation)
            helicity_embed = nn.Sequential(*hel_layers)

            head = []
            head.append(
                nn.Linear(
                    self.cfg.model["internal_size"]
                    + self.cfg.dataset.embed_helicities.get("embed_dimension", 64),
                    self.cfg.model["internal_size"],
                )
            )
            head.append(activation)
            for _ in range(self.cfg.model["hidden_layers"]):
                head.append(
                    nn.Linear(
                        self.cfg.model["internal_size"], self.cfg.model["internal_size"]
                    )
                )
                head.append(activation)
            head.append(nn.Linear(self.cfg.model["internal_size"], self.dims_out))
            head_net = nn.Sequential(*head)

            self.net = nn.ModuleDict(
                {
                    "feature_embed": feature_embed,
                    "helicity_embed": helicity_embed,
                    "head_net": head_net,
                }
            )
        else:
            layers = []
            layers.append(nn.Linear(self.dims_in, self.cfg.model["internal_size"]))
            layers.append(activation)
            for _ in range(self.cfg.model["hidden_layers"]):
                layers.append(
                    nn.Linear(
                        self.cfg.model["internal_size"], self.cfg.model["internal_size"]
                    )
                )
                layers.append(activation)
            layers.append(nn.Linear(self.cfg.model["internal_size"], self.dims_out))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.cfg.dataset.embed_helicities.get("use", False):
            features, helicities = x[:, :-1], x[:, -1].long()
            feat_embed = self.net["feature_embed"](features)
            hel_embed = self.net["helicity_embed"](helicities)
            cat = torch.cat([feat_embed, hel_embed], dim=-1)
            return self.net["head_net"](cat)
        else:
            return self.net(x)

    def predict(self, x):
        return self.forward(x)


class Transformer(Model):
    def __init__(
        self,
        logger,
        process,
        cfg,
        dims_in,
        helicity_dict_size,
        dims_out,
        model_path,
        device,
    ):
        super().__init__(logger, cfg, dims_in, dims_out, model_path, device)
        assert (
            cfg.dataset.parameterization.naive.use
        ), "Only naive parameterization is supported for the Transformer model"
        self.remove_color = cfg.dataset.get("remove_color", False)
        self.init_loss()
        self.init_net()

    def init_net(self):
        self.dim_embedding = self.cfg.model["dim_embedding"]
        if self.cfg.model.get("activation", "gelu") == "relu":
            self.activation = nn.ReLU()
        elif self.cfg.model.get("activation", "gelu") == "silu":
            self.activation = nn.SiLU()
        elif self.cfg.model.get("activation", "gelu") == "gelu":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError(
                f"Activation function {self.cfg.model.get('activation', 'gelu')} not implemented"
            )
        self.logger.info(
            f"    Using {self.activation} activation function for Transformer model"
        )
        self.features_per_particle = 7 if not self.remove_color else 6
        self.n_particles = self.dims_in // self.features_per_particle
        input_dim = self.n_particles + self.features_per_particle
        self.input_proj = nn.Linear(input_dim, self.dim_embedding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_embedding,
            nhead=self.cfg.model.get("n_head", 8),
            dim_feedforward=self.cfg.model.get("dim_feedforward", 512),
            dropout=self.cfg.model.get("dropout", 0.1),
            activation=self.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.cfg.model.get("n_layers", 6)
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.dim_embedding, 2 * self.dim_embedding),
            self.activation,
            nn.Linear(2 * self.dim_embedding, self.dim_embedding),
            self.activation,
            nn.Linear(self.dim_embedding, self.dims_out),
        )
        self.net = nn.Sequential(self.input_proj, self.encoder, self.regressor)

    def forward(self, x):
        n_particles = x.shape[1] // self.features_per_particle
        x = x.view(-1, n_particles, self.features_per_particle)
        # One hot encoding
        one_hot = torch.eye(n_particles, device=x.device)
        one_hot = one_hot.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([x, one_hot], dim=-1)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.sum(dim=1) / n_particles
        x = self.regressor(x)
        return x

    def predict(self, x):
        return self.forward(x)


class TransformerExtrapolator(Model):
    def __init__(
        self,
        logger,
        process,
        cfg,
        dims_in,
        helicity_dict_size,
        dims_out,
        model_path,
        device,
    ):
        super().__init__(logger, cfg, dims_in, dims_out, model_path, device)
        assert (
            cfg.dataset.parameterization.naive.use
        ), "Only naive parameterization is supported for the Transformer model"
        self.remove_color = cfg.dataset.get("remove_color", False)
        self.init_loss()
        self.init_net()

    def init_net(self):
        self.dim_embedding = self.cfg.model["dim_embedding"]
        if self.cfg.model.get("activation", "gelu") == "relu":
            self.activation = nn.ReLU()
        elif self.cfg.model.get("activation", "gelu") == "silu":
            self.activation = nn.SiLU()
        elif self.cfg.model.get("activation", "gelu") == "gelu":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError(
                f"Activation function {self.cfg.model.get('activation', 'gelu')} not implemented"
            )
        self.logger.info(
            f"    Using {self.activation} activation function for Transformer model"
        )
        self.max_n_particles = 9  # Set this globally
        self.features_per_particle = 7 if not self.remove_color else 6
        input_dim = self.features_per_particle + self.max_n_particles
        self.input_proj = nn.Linear(input_dim, self.dim_embedding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_embedding,
            nhead=self.cfg.model.get("n_head", 8),
            dim_feedforward=self.cfg.model.get("dim_feedforward", 512),
            dropout=self.cfg.model.get("dropout", 0.1),
            activation=self.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.cfg.model.get("n_layers", 6)
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.dim_embedding, 2 * self.dim_embedding),
            self.activation,
            nn.Linear(2 * self.dim_embedding, self.dim_embedding),
            self.activation,
            nn.Linear(self.dim_embedding, self.dims_out),
        )
        self.net = nn.Sequential(self.input_proj, self.encoder, self.regressor)

    def forward(self, x):
        batch_size = x.shape[0]
        features_per_particle = self.features_per_particle
        max_n_particles = self.max_n_particles

        n_particles = x.shape[1] // features_per_particle

        # Pad if needed
        if n_particles < max_n_particles:
            pad_size = max_n_particles - n_particles
            x = x.view(batch_size, n_particles, features_per_particle)
            pad = torch.zeros(
                batch_size,
                pad_size,
                features_per_particle,
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)
        else:
            x = x.view(batch_size, n_particles, features_per_particle)

        # One-hot for slot identity
        one_hot = (
            torch.eye(max_n_particles, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        x = torch.cat([x, one_hot], dim=-1)
        x = self.input_proj(x)
        x = self.encoder(x)

        # Create mask: 1 for real, 0 for padded --> only real particles contribute to the mean
        mask = torch.zeros(batch_size, max_n_particles, device=x.device, dtype=x.dtype)
        mask[:, :n_particles] = 1.0
        mask = mask.unsqueeze(-1)

        x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        x = self.regressor(x)
        return x

    def predict(self, x):
        return self.forward(x)
