import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.mlflow import mlflow, LOGGING_ENABLED
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric


class Model(nn.Module):
    def __init__(self, logger, cfg, dims_in, dims_out, _shift, model_path, device):
        super().__init__()
        self.logger = logger
        self.cfg = cfg
        self.dims_in = dims_in
        self.dims_out = dims_out
        self._shift = _shift
        self.model_path = model_path
        self.device = device
        self.heteroscedastic_loss = cfg.train.get("heteroscedastic_loss", None)

        self.state_dict_attrs = ["net", "optimizer"]  # add more if needed
        if self.cfg.train.scheduler is not None:
            self.state_dict_attrs.append("scheduler")

    # Overwrite in child class
    def init_net(self):
        pass

    # Overwrite in child class
    def sample(self, c):
        pass

    # Overwrite in child class
    def predict(self, x):
        pass

    # Overwrite in child class
    def forward(self, x):
        pass

    # Overwrite in child class
    def batch_loss(self, x, y, weight):
        pass

    def init_dataloaders(self, dataset, weights=None):
        data_ppd = dataset.events_ppd
        channels = dataset.channels

        # apply stuff to make it such that only the channels of the data are used
        if weights is None:
            weights = {
                f"{split}": torch.ones(data_ppd[f"{split}"].shape[0], 1)
                .to(data_ppd[f"{split}"].dtype)
                .to(data_ppd[f"{split}"].device)
                for split in ["trn", "tst", "val"]
            }
        for split in ["trn", "tst", "val"]:
            globals()[f"{split}set"] = torch.utils.data.TensorDataset(
                data_ppd[f"{split}"][..., channels],
                weights[f"{split}"],
            )
        self.trnloader = torch.utils.data.DataLoader(
            trnset, batch_size=self.cfg.train.batch_size, shuffle=True
        )
        self.valloader = torch.utils.data.DataLoader(
            valset, batch_size=self.cfg.train.batch_size, shuffle=True
        )
        self.tstloader = torch.utils.data.DataLoader(
            tstset,
            batch_size=self.cfg.train.get("batch_size_eval", self.cfg.train.batch_size),
            shuffle=False,
        )

    def init_optimizer(self):
        optim = self.cfg.train.get("optimizer", "adam")
        lr = float(self.cfg.train.get("lr", 0.001))
        wd = self.cfg.train.get("weight_decay", 0.0)
        if optim == "adam" or optim == "Adam":
            optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=wd)
        else:
            raise NotImplementedError(f"Optimizer {optim} not implemented")
        self.optimizer = optimizer
        self.logger.info(f"    Using optimizer {optim} with lr={lr} and weight decay={wd}")

    def init_scheduler(self):
        sched = self.cfg.train.get("scheduler", None)
        if sched == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, len(self.trnloader) * self.cfg.train["nepochs"]
            )
        elif sched == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.cfg.train.get("lr_factor", 0.1),
                patience=self.cfg.train.get("lr_patience", 10),
            )
        elif sched is None:
            scheduler = None
        else:
            raise NotImplementedError(f"Scheduler {sched} not implemented")
        self.scheduler = scheduler
        self.logger.info(f"    Using scheduler {sched}")

    def train(self):
        if LOGGING_ENABLED:
            self.mlflowclient = MlflowClient()
        self.net.train()
        nepochs = self.cfg.train.nepochs
        self.init_optimizer()
        self.init_scheduler()
        if self.cfg.train.get("warm_start", False):
            self.logger.info(
                "Warm-starting model from "
                + os.path.join(
                    self.model_path, f"{self.cfg.get('checkpoint', 'final')}.pth"
                )
            )
            self.load(self.cfg.train.get("checkpoint", "final"))
        if self.heteroscedastic_loss is not None and self.heteroscedastic_loss.use:
            hs_loss = []
            self.first_hs_call = True
        else:
            hs_loss = None
        trn_loss = []
        val_loss = []
        trn_lr = []
        grd_norm = []
        self.best_val_loss = 1e20
        if (
            self.heteroscedastic_loss.use
            and self.heteroscedastic_loss.get("activate_after_its", 1000) > 1
        ):
            self.logger.info(
                f"Will use heteroscedastic loss with scale {self.heteroscedastic_loss.scale} after {round(self.heteroscedastic_loss.get('activate_after_its', 1000))} iterations"
            )
        elif self.heteroscedastic_loss.use:
            self.logger.info(
                f"Will use heteroscedastic loss with scale {self.heteroscedastic_loss.scale} after {round(len(self.trnloader) * self.cfg.train.nepochs * self.heteroscedastic_loss.get('activate_after_its', 1000))} iterations"
            )
        else:
            pass
        self.logger.info(
            f"Training model for {nepochs} epochs (= {nepochs * len(self.trnloader)} iterations)"
        )
        t0 = time.time()
        for epoch in range(1, nepochs + 1):
            epoch_val_losses = []
            for i, batch in enumerate(self.trnloader):
                x, weight = batch
                self.optimizer.zero_grad()
                pred = self.forward(x[:, :-3])
                target = x[:, self._shift].unsqueeze(-1)
                current_it = 1 + (epoch - 1) * len(self.trnloader) + i
                loss, loss_terms = self.batch_loss(pred, target, weight, current_it)
                loss.backward()
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(),
                        self.cfg.train.get("clip_grad_norm", 10),
                        error_if_nonfinite=False,
                    )
                    .cpu()
                    .item()
                )
                self.optimizer.step()
                if self.scheduler is not None and not isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()
                trn_loss.append(loss.cpu().item())
                if "hs_loss" in loss_terms:
                    hs_loss.append(loss_terms["hs_loss"].cpu().item())
                trn_lr.append(self.optimizer.param_groups[0]["lr"])
                grd_norm.append(grad_norm)

            for i, batch in enumerate(self.valloader):
                with torch.no_grad():
                    x, weight = batch
                    pred = self.forward(x[:, :-3])
                    target = x[:, self._shift].unsqueeze(-1)
                    loss, loss_terms = self.batch_loss(
                        pred,
                        target,
                        weight,
                        current_it=self.cfg.train.nepochs * len(self.trnloader),
                        val=True,
                    )
                    epoch_val_losses.append(loss.cpu().item())

            avg_trn_loss = torch.tensor(trn_loss).mean().item()
            avg_val_loss = torch.tensor(epoch_val_losses).mean().item()
            if self.scheduler is not None and isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(avg_val_loss)
            val_loss.append(avg_val_loss)
            self.losses = {
                "trn": trn_loss,
                "val": val_loss,
                "lr": trn_lr,
                "grad_norm": grd_norm,
            }
            if self.heteroscedastic_loss.use:
                self.losses["hs"] = hs_loss
                self.losses["hs_scale"] = loss_terms["hs_scale"]

            self.log_and_save(
                t0,
                epoch,
                avg_trn_loss,
                avg_val_loss,
            )
        self.save("final")
        self.logger.info(
            f"Finished training after {time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))}"
        )
        self.logger.info(f"Best validation loss: {self.best_val_loss:.8f}")
        self.logger.info(f"Last validation loss: {self.losses['val'][-1]:.8f}")


    def log_and_save(self, t0, epoch, avg_trn_loss, avg_val_loss):
        if epoch == 1:
            self.logger.info(
                f"    Epoch {epoch}: tr_loss={avg_trn_loss:.8f}, val_loss={avg_val_loss:.8f}; ETA={time.strftime('%H:%M:%S', time.gmtime((time.time() - t0) * self.cfg.train.nepochs))}"
            )
            
        else:
            if LOGGING_ENABLED and epoch % max(1, self.cfg.mlflow.get("log_every", 1)) == 0:
                if mlflow.active_run() is None:
                    self.logger.warning(
                        "MLflow is not active. Cannot log metrics."
                    )
                else:
                    ts = int((time.time()-t0) * 1000)
                
                    metrics_batch = []
                    for step_idx, loss_val in enumerate(self.losses["trn"]):
                        metrics_batch.append(
                            {
                                "key":       "trn_loss_per_iter",
                                "value":     loss_val,
                                "timestamp": ts,
                                "step":      step_idx,
                            }
                        )
                    metrics_batch.append(
                        {
                            "key":       "val_loss",
                            "value":     self.losses["val"][-1],
                            "timestamp": ts,
                            "step":      epoch,
                        }
                    )
                    metrics_batch.append(
                        {
                            "key":       "lr",
                            "value":     self.losses["lr"][-1],
                            "timestamp": ts,
                            "step":      epoch,
                        }
                    )
                    metrics_batch.append(
                        {
                            "key":       "grad_norm",
                            "value":     self.losses["grad_norm"][-1],
                            "timestamp": ts,
                            "step":      epoch,
                        }
                    )
                    metrics = [
                        Metric(
                            key=m["key"],
                            value=m["value"],
                            timestamp=m["timestamp"],
                            step=m["step"],
                        )
                        for m in metrics_batch
                    ]

                    if self.heteroscedastic_loss.use:
                        self.logger.warning(
                            "Heteroscedastic logging is not supported yet"
                        )
                    self.mlflowclient.log_batch(run_id=mlflow.active_run().info.run_id, metrics=metrics)
            log_every_percent = 0.10
            if epoch % max(1, int(self.cfg.train.nepochs * log_every_percent)) == 0:
                self.logger.info(
                    f"    Epoch {epoch}: tr_loss={avg_trn_loss:.8f}, val_loss={avg_val_loss:.8f}"
                )
            if self.losses["val"][-1] < self.best_val_loss:
                self.best_val_loss = self.losses["val"][-1]
                if epoch > 10:
                    self.save("best")
                    self.logger.info(
                        f"    Saved best model with val_loss={self.losses['val'][-1]:.8f} at epoch {epoch}"
                    )


    def save(self, name: str):
        """
        Saves the model, preprocessing, optimizer and losses.

        Args:
            name: File name for the model (without path and extension)
        """
        file = os.path.join(self.model_path, f"{name}.pth")
        torch.save(
            {
                **{
                    attr: getattr(self, attr).state_dict()
                    for attr in self.state_dict_attrs
                },
                "losses": self.losses,
            },
            file,
        )

    def load(self, name: str):
        """
        Loads the model, preprocessing, optimizer and losses.

        Args:
            name: File name for the model (without path and extension)
        """
        file = os.path.join(self.model_path, f"{name}.pth")
        state_dicts = torch.load(file, map_location=self.device)
        for attr in self.state_dict_attrs:
            try:
                getattr(self, attr).load_state_dict(state_dicts[attr])
            except AttributeError:
                pass
        self.losses = state_dicts["losses"]

    def evaluate(self, loader=None):
        if loader is None:
            loader = self.tstloader
        if loader is self.tstloader:
            self.logger.info("Evaluating model on tst set")
        elif loader is self.valloader:
            self.logger.info("Evaluating model on val set")
        else:
            self.logger.info("Evaluating model on trn set")
        predictions = []
        self.net.eval()
        with torch.no_grad():
            t0 = time.time()
            for i, batch in enumerate(loader):
                x, weight = batch
                predicted_factors = self.predict(x[:, :-3]).detach().cpu()
                predictions.append(predicted_factors)
                t1 = time.time()
                if i == 0:
                    self.logger.info(
                        f"    Total batches: {len(loader)}. Sampling time estimate: {time.strftime('%H:%M:%S', time.gmtime(round((t1-t0) * len(loader), 1)))}"
                    )
                log_every_percent = 0.25
                if i % max(1, int(len(loader) * log_every_percent)) == 0:
                    self.logger.info(f"    Sampled batch {i+1}/{len(loader)}")
            self.logger.info(f"    Finished sampling. Saving predictions")
        predictions = torch.cat(predictions)
        return predictions

    def batch_loss(self, pred, target, weight, current_it=None, val=False, debug=False):
        if debug:
            print(pred, target)
        regression_loss = self.loss_fct(pred, target)

        # heteroscedastic loss
        activate_hs_loss = (
            current_it > self.heteroscedastic_loss.get("activate_after_its", 1000)
            if self.heteroscedastic_loss.get("activate_after_its", 1000) > 1
            else current_it / len(self.trnloader) / self.cfg.train.nepochs
            > self.heteroscedastic_loss.get("activate_after_its", 1000)
        )

        if self.heteroscedastic_loss.use and activate_hs_loss:
            if self.first_hs_call and not val:
                self.logger.info(
                    f"    Using heteroscedastic loss with scale {self.cfg.train.heteroscedastic_loss.get('scale', 0.001)} from it. {current_it} onwards"
                )
                self.first_hs_call = False
            hs_loss = (
                self.cfg.train.heteroscedastic_loss.get("scale", 0.001)
                * (target / pred).std()
            )

        else:
            hs_loss = torch.tensor([0.0]).to(pred.device)
        loss = regression_loss + hs_loss.mean()

        loss_terms = {
            "loss": loss,
            "reg_loss": regression_loss,
        }
        if self.heteroscedastic_loss.use:
            loss_terms["hs_loss"] = hs_loss
            loss_terms["hs_scale"] = self.cfg.train.heteroscedastic_loss.get(
                "scale", 0.001
            )
        return loss, loss_terms


class MLP(Model):
    def __init__(self, logger, process, cfg, dims_in, dims_out, _shift, model_path, device):
        super().__init__(logger, cfg, dims_in, dims_out, _shift, model_path, device)
        self.loss_fct = (
            nn.L1Loss() if self.cfg.train.get("loss", "MSE") == "L1" else nn.MSELoss()
        )
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
        return self.net(x)

    def predict(self, x):
        return self.forward(x)


class Transformer(Model):
    def __init__(self, logger, process, cfg, dims_in, dims_out, _shift, model_path, device):
        super().__init__(logger, cfg, dims_in, dims_out, _shift, model_path, device)
        self.loss_fct = (
            nn.L1Loss() if cfg.train.get("loss", "MSE") == "L1" else nn.MSELoss()
        )
        assert (
            cfg.dataset.parameterisation.naive.use
        ), "Only naive parameterisation is supported for the Transformer model"
        self.init_net()

    def init_net(self):
        self.dim_embedding = self.cfg.model["dim_embedding"]
        self.n_particles = self.dims_in // 4
        input_dim = 4 + self.n_particles
        self.input_proj = nn.Linear(input_dim, self.dim_embedding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_embedding,
            nhead=self.cfg.model.get("n_head", 8),
            dim_feedforward=self.cfg.model.get("dim_feedforward", 512),
            dropout=self.cfg.model.get("dropout", 0.1),
            activation=self.cfg.model.get("activation", "gelu"),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.cfg.model.get("n_layers", 6)
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.dim_embedding, self.dim_embedding),
            nn.ReLU()
            if self.cfg.model.get("activation", "gelu") == "relu"
            else nn.SiLU()
            if self.cfg.model.get("activation", "gelu") == "SiLU"
            else nn.GELU(),
            nn.Linear(self.dim_embedding, 1),
        )
        self.net = nn.Sequential(self.input_proj, self.encoder, self.regressor)

    def forward(self, x):
        n_particles = x.shape[1] // 4
        x = x.view(-1, n_particles, 4)

        # One hot encoding
        one_hot = torch.eye(self.n_particles, device=x.device)
        one_hot = one_hot.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([x, one_hot], dim=-1)

        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.sum(dim=1) / n_particles
        x = self.regressor(x)
        return x

    def predict(self, x):
        return self.forward(x)
