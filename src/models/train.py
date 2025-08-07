import time
import os
import numpy as np
import torch
import torch.nn as nn
from src.utils.mlflow import mlflow, LOGGING_ENABLED
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric
import src.utils.physics as phys
from src.plots import *
from src.utils.plots_utils import Metric as DataMetric


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Model(nn.Module):
    def __init__(self, logger, cfg, dims_in, dims_out, model_path, device):
        super().__init__()
        self.logger = logger
        self.cfg = cfg
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.model_path = model_path
        self.device = device

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
        channels = dataset.input_channels

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
                data_ppd[f"{split}"][..., channels].to(torch.float64),
                weights[f"{split}"].to(torch.float64),
            )
        self.trnloader = torch.utils.data.DataLoader(
            trnset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
        )
        self.valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
        )
        self.tstloader = torch.utils.data.DataLoader(
            tstset,
            batch_size=self.cfg.train.get("batch_size_eval", self.cfg.train.batch_size),
            shuffle=False,
        )
        self.inf_trnloader = torch.utils.data.DataLoader(
            trnset,
            batch_size=self.cfg.train.get("batch_size_eval", self.cfg.train.batch_size),
            shuffle=False,
        )
        self.inf_valloader = torch.utils.data.DataLoader(
            valset,
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
        self.logger.info(
            f"    Using optimizer {optim} with lr={lr} and weight decay={wd}"
        )

    def init_scheduler(self):
        sched = self.cfg.train.get("scheduler", None)
        if sched == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.cfg.train.nits
            )
        elif sched == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.cfg.train.get("lr_factor", 0.1),
                patience=self.cfg.train.get(
                    "lr_patience", 10
                ),  # nb specified in epochs, but we use iterations
            )
        elif sched is None:
            scheduler = None
        else:
            raise NotImplementedError(f"Scheduler {sched} not implemented")
        self.scheduler = scheduler
        self.logger.info(f"    Using scheduler {sched}") if not hasattr(
            self.scheduler, "patience"
        ) else self.logger.info(
            f"    Using scheduler {sched} with factor {getattr(self.scheduler, 'factor')} and patience {getattr(self.scheduler, 'patience')}."
        )

    def train(self):
        if LOGGING_ENABLED:
            self.mlflowclient = MlflowClient()
        nits = self.cfg.train.nits  # Number of training iterations (steps)
        val_freq = self.cfg.train.get(
            "val_freq", len(self.trnloader)
        )  # How often to validate/save, default: per epoch
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
        self.best_val_loss = 1e20
        if self.cfg.train.early_stopping.get("use", False):
            patience = self.cfg.train.early_stopping.get("patience", 10)
            early_stopping = EarlyStopping(patience=patience)
            self.logger.info(f"    Using early stopping with patience {patience}")
        self.logger.info(
            f"Training model for {nits} iterations (= {nits // len(self.trnloader)} epochs)"
        )
        self.logger.info(
            f"    Validation frequency: {val_freq} its. Nb of train batches: {len(self.trnloader)}"
        )

        current_it = 0
        epoch = 0
        t0 = time.time()
        trn_loss = []
        epoch_avg_val_loss = []
        trn_lr = []
        grd_norm = []
        stop_training = False
        while current_it < nits and not stop_training:
            epoch += 1
            epoch_avg_trn_loss = []
            # check if model is in training mode
            self.net.train()
            for i, batch in enumerate(self.trnloader):
                if current_it >= nits:
                    break
                x, weight = batch
                self.optimizer.zero_grad()
                pred = self.forward(x[:, :-1])
                target = x[:, -1].unsqueeze(-1)
                loss, loss_terms = self.batch_loss(pred, target, weight)
                loss = loss.mean()
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
                epoch_avg_trn_loss.append(loss.cpu().item())
                trn_lr.append(self.optimizer.param_groups[0]["lr"])
                grd_norm.append(grad_norm)
                current_it += 1

                if current_it % val_freq == 0 or current_it == nits:
                    avg_val_loss = self.validate(
                        t0=t0,
                        current_it=current_it,
                        trn_loss=trn_loss,
                        epoch_avg_trn_loss=epoch_avg_trn_loss,
                        epoch_avg_val_loss=epoch_avg_val_loss,
                        trn_lr=trn_lr,
                        grd_norm=grd_norm,
                    )
                    if self.scheduler is not None and isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step(avg_val_loss)
                    if self.cfg.train.early_stopping.get("use", False):
                        early_stopping(avg_val_loss)
                        if early_stopping.early_stop:
                            self.logger.info(
                                f"Early stopping after iteration {current_it} with validation loss {avg_val_loss:.8f}"
                            )
                            stop_training = True
                            break
        self.save("final")
        self.logger.info(
            f"Finished training after {time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))}. Nb of its: {current_it}, epochs: {epoch}"
        )
        self.logger.info(f"Best validation loss: {self.best_val_loss:.8f}")
        self.logger.info(f"Last validation loss: {self.losses['val'][-1]:.8f}")

    def validate(
        self,
        t0,
        current_it,
        trn_loss,
        epoch_avg_trn_loss,
        epoch_avg_val_loss,
        trn_lr,
        grd_norm,
    ):
        val_losses = []
        for j, vbatch in enumerate(self.valloader):
            with torch.no_grad():
                vx, vweight = vbatch
                vpred = self.forward(vx[:, :-1])
                vtarget = vx[:, -1].unsqueeze(-1)
                vloss, vloss_terms = self.batch_loss(
                    vpred,
                    vtarget,
                    vweight,
                )
                vloss = vloss.mean()
                val_losses.append(vloss.cpu().item())
        epoch_avg_trn_loss = torch.tensor(epoch_avg_trn_loss).mean().item()
        avg_val_loss = torch.tensor(val_losses).mean().item()
        epoch_avg_val_loss.append(avg_val_loss)
        self.losses = {
            "trn": trn_loss,
            "val": epoch_avg_val_loss,
            "lr": trn_lr,
            "grad_norm": grd_norm,
        }
        self.log_and_save(t0, current_it, epoch_avg_trn_loss, avg_val_loss)
        return avg_val_loss

    def log_and_save(self, t0, iteration, avg_trn_loss, avg_val_loss):
        if iteration == 1 * len(self.trnloader):
            self.logger.info(
                f"    Epoch {iteration // len(self.trnloader)} (it. {iteration}): tr_loss={avg_trn_loss:.8f}, val_loss={avg_val_loss:.8f}; ETA={time.strftime('%H:%M:%S', time.gmtime((time.time() - t0) * (self.cfg.train.nits - iteration) / iteration))}"
            )
        if (
            LOGGING_ENABLED
            and iteration
            % max(
                1,
                int(0.10 * self.cfg.train.nits // len(self.trnloader))
                * self.cfg.train.get("val_freq", len(self.trnloader)),
            )
            == 0
        ):
            self.mlflow_log_metrics(t0, iteration)

        # log_every_percent = 0.10
        # if iteration % max(1, int(log_every_percent * self.cfg.train.nits // len(self.trnloader))*self.cfg.train.get("val_freq", len(self.trnloader))) == 0:
        if (
            iteration
            % max(
                1,
                # iteration
                int(0.001 * self.cfg.train.nits // len(self.trnloader))
                * self.cfg.train.get("val_freq", len(self.trnloader)),
            )
            == 0
        ):
            self.logger.info(
                f"    Epoch {iteration // len(self.trnloader)} (it. {iteration}) : tr_loss={avg_trn_loss:.8f}, val_loss={avg_val_loss:.8f}"
            )

            # if self.cfg.train.get("plot_preds_vs_targets", True):
            #     # self.logger.info(
            #     #     f"    Plotting predictions vs targets for it. {iteration}"
            #     # )
            #     self.plot_predictions_vs_targets_at_train(iteration=iteration)
            #     if iteration > 5800 and iteration < 5900 or iteration > 10200 and iteration < 10300 or iteration > 20200 and iteration < 20300 or iteration > 30100 and iteration < 30300 or iteration > 40100 and iteration < 40300 or iteration > 49900:
            #         self.save(f"it_{iteration}")
                
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            if iteration > 10 * len(self.trnloader):  # Avoid saving too early
                self.save("best")
                self.logger.info(
                    f"    Saved best model with val_loss={avg_val_loss:.8f} at it. {iteration}"
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

    def evaluate(
        self,
        split=None,
    ):
        if split is None or split == "tst":
            loader = self.tstloader
            self.logger.info("Evaluating model on tst set")
        elif split == "val":
            loader = self.inf_valloader
            self.logger.info("Evaluating model on val set")
        else:
            loader = self.inf_trnloader
            self.logger.info("Evaluating model on trn set")
        predictions = []
        self.net.eval()
        losses = []
        with torch.no_grad():
            t0 = time.time()
            for i, batch in enumerate(loader):
                x, weight = batch
                target = x[:, -1].unsqueeze(-1)
                pred = self(x[:, :-1])
                losses.append(
                    self.batch_loss(
                        pred,
                        target,
                        weight,
                    )[0]
                    .detach()
                    .cpu()
                )
                predictions.append(pred.squeeze().detach().cpu()) if self.cfg.train.get(
                    "loss", "MSE"
                ) != "heteroschedastic" else predictions.append(
                    pred[..., 0].squeeze().detach().cpu()
                )
                t1 = time.time()
                if i == 0:
                    self.logger.info(
                        f"    Total batches: {len(loader)}. Sampling time estimate: {time.strftime('%H:%M:%S', time.gmtime(round((t1-t0) * len(loader), 1)))}"
                    )
                log_every_percent = 0.25
                if (
                    i % max(1, int(len(loader) * log_every_percent)) == 0
                ):
                    self.logger.info(f"    Sampled batch {i+1}/{len(loader)}")
        self.logger.info(f"    Finished sampling in {time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))}. Saving predictions")
        predictions = torch.cat(predictions)
        dataset_loss = torch.cat(losses).mean().item()
        # self.logger.info(f"Loss on {split} (ppd) set: {dataset_loss:.3e}")
        self.dataset_loss["ppd"][split] = dataset_loss
        return predictions

    def evaluate_in_training(
        self,
        split=None,
        during_training=False,
        boost=False,
        SO3=False,
        SL4=False,
        shear=False,
        SO2=False,
    ):

        if split is None or split == "tst":
            loader = self.tstloader
            self.logger.info("Evaluating model on tst set")
        elif split == "val":
            loader = self.inf_valloader
            if not during_training:
                self.logger.info("Evaluating model on val set")
        else:
            loader = self.inf_trnloader
            self.logger.info("Evaluating model on trn set")
        predictions = []
        self.net.eval()
        losses = []
        targets = [] if during_training else None
        with torch.no_grad():
            t0 = time.time()
            for i, batch in enumerate(loader):
                x, weight = batch
                target = x[:, -1].unsqueeze(-1)
                batch_size = x.shape[0]
                if boost:
                    assert during_training
                    # boost_matrix, boost_matrix_inv = phys.random_lorentz_boost(
                    #     beta=torch.tensor(0.5), # beta kept between 0 and 0.8 to avoid numerical instabilities close to 1
                    #     device=x.device,
                    # )
                    # x[:, :-1] = phys.apply_lorentz_boost_to_tensor(x[:, :-1], boost_matrix=boost_matrix, boost_inv_matrix=boost_matrix_inv)
                    batch_size = x.shape[0]
                    boost_matrices = phys.batch_random_lorentz_boost(
                        batch_size, device=x.device, dtype=torch.float64
                    )
                    x[:, :-1] = phys.apply_rotation_to_tensor_vectorized(
                        x[:, :-1], rotation_matrices=boost_matrices
                    )

                elif SO3:
                    assert during_training
                    # rotation_matrix = phys.random_SO3_matrix(
                    #     device=x.device, dtype=torch.float64
                    # )
                    # x[:, :-1] = phys.apply_rotation_to_tensor(x[:, :-1], rotation_matrix)
                    SO3_matrices = phys.batch_random_SO3_matrix(
                        batch_size, device=x.device, dtype=torch.float64
                    )
                    x[:, :-1] = phys.apply_rotation_to_tensor_vectorized(
                        x[:, :-1], rotation_matrices=SO3_matrices
                    )
                elif SL4:
                    assert during_training
                    # rotation_matrix = phys.random_SL4_matrix(
                    #     device=x.device, dtype=torch.float64
                    # )
                    # x[:, :-1] = phys.apply_rotation_to_tensor(x[:, :-1], rotation_matrix)
                    SL4_matrices = phys.batch_random_SL4_matrix(
                        batch_size, device=x.device, dtype=torch.float64
                    )
                    x[:, :-1] = phys.apply_rotation_to_tensor_vectorized(
                        x[:, :-1], rotation_matrices=SL4_matrices
                    )
                elif shear:
                    assert during_training
                    # shear_matrix = phys.random_shear_matrix(
                    #     device=x.device, dtype=torch.float64
                    # )
                    # x[:, :-1] = phys.apply_rotation_to_tensor(x[:, :-1], shear_matrix)
                    shear_matrices = phys.batch_random_shear_matrix(
                        batch_size, device=x.device, dtype=torch.float64
                    )
                    x[:, :-1] = phys.apply_rotation_to_tensor_vectorized(
                        x[:, :-1], rotation_matrices=shear_matrices
                    )
                elif SO2:
                    assert during_training
                    # rotation_matrix = phys.random_SO2_matrix(
                    #     device=x.device, dtype=torch.float64
                    # )
                    # x[:, :-1] = phys.apply_rotation_to_tensor(x[:, :-1], rotation_matrix)
                    rotation_matrices = phys.batch_random_SO2_matrix(
                        batch_size, device=x.device, dtype=torch.float64
                    )
                    x[:, :-1] = phys.apply_rotation_to_tensor_vectorized(
                        x[:, :-1], rotation_matrices=rotation_matrices
                    )
                pred = self.predict(x[:, :-1])
                if during_training:
                    targets.append(target.squeeze().detach().cpu())
                losses.append(
                    self.batch_loss(
                        pred,
                        target,
                        weight,
                    )[0]
                    .detach()
                    .cpu()
                )
                predictions.append(pred.squeeze().detach().cpu()) if self.cfg.train.get(
                    "loss", "MSE"
                ) != "heteroschedastic" else predictions.append(
                    pred[..., 0].squeeze().detach().cpu()
                )
                t1 = time.time()
                if i == 0 and not during_training:
                    self.logger.info(
                        f"    Total batches: {len(loader)}. Sampling time estimate: {time.strftime('%H:%M:%S', time.gmtime(round((t1-t0) * len(loader), 1)))}"
                    )
                log_every_percent = 0.25
                if (
                    i % max(1, int(len(loader) * log_every_percent)) == 0
                    and not during_training
                ):
                    self.logger.info(f"    Sampled batch {i+1}/{len(loader)}")
        if not during_training:
            self.logger.info(f"    Finished sampling. Saving predictions")
        predictions = torch.cat(predictions)
        if targets is not None:
            targets = torch.cat(targets)
        dataset_loss = torch.cat(losses).mean().item()
        if not during_training:
            self.logger.info(f"Loss on {split} (ppd) set: {dataset_loss:.3e}")
            self.dataset_loss["ppd"][split] = dataset_loss
        if targets is not None:
            return predictions, targets, (dataset_loss, torch.cat(losses).median().item())
        else:
            return predictions

    def compute_dataset_loss(self, raw_predictions, raw_targets, split=None):
        if not hasattr(self, "dataset_loss"):
            self.dataset_loss = {}
        loss, _ = self.batch_loss(
            raw_predictions,
            raw_targets,
            weight=None,
            debug=self.cfg.train.get("debug", False),
        )
        loss = loss.mean().item()
        self.logger.info(f"Loss on {split} (raw) set: {loss:.3e}")
        self.dataset_loss["raw"][split] = loss

    def batch_loss(self, pred, target, weight, debug=False):
        if debug:
            print(pred.shape, target.shape)
        regression_loss = self.loss_fct(pred, target)
        loss = regression_loss

        loss_terms = {
            "loss": loss,
            "reg_loss": regression_loss,
        }
        return loss, loss_terms

    def mlflow_log_metrics(self, t0, iteration):
        if mlflow.active_run() is None:
            self.logger.warning("MLflow is not active. Cannot log metrics.")
        else:
            ts = int((time.time() - t0) * 1000)
            metrics_batch = []
            # for step_idx, loss_val in enumerate(self.losses["trn"]):
            #     metrics_batch.append(
            #         {
            #             "key":       "trn_loss_per_iter",
            #             "value":     loss_val,
            #             "timestamp": ts,
            #             "step":      step_idx,
            #         }
            #     )
            metrics_batch.append(
                {
                    "key": "trn_loss",
                    "value": self.losses["trn"][-1],
                    "timestamp": ts,
                    "step": iteration,
                }
            )
            metrics_batch.append(
                {
                    "key": "val_loss",
                    "value": self.losses["val"][-1],
                    "timestamp": ts,
                    "step": iteration,
                }
            )
            metrics_batch.append(
                {
                    "key": "lr",
                    "value": self.losses["lr"][-1],
                    "timestamp": ts,
                    "step": iteration,
                }
            )
            metrics_batch.append(
                {
                    "key": "grad_norm",
                    "value": self.losses["grad_norm"][-1],
                    "timestamp": ts,
                    "step": iteration,
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
            self.mlflowclient.log_batch(
                run_id=mlflow.active_run().info.run_id, metrics=metrics
            )

    def init_loss(self):
        loss_type = self.cfg.train.get("loss", "MSE")
        if loss_type == "MSE":
            self.loss_fct = nn.MSELoss(reduction="none")
        elif loss_type == "L1":
            self.loss_fct = nn.L1Loss(reduction="none")
        elif loss_type == "heteroschedastic":
            self.loss_fct = self.het_loss
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented")

    def het_loss(self, pred, target):
        mu, logsigma2 = pred[..., 0:1], pred[..., 1:2]
        logsigma2 = logsigma2.clamp(-30, 11)
        sigma2 = logsigma2.exp()
        reco = (target - mu) ** 2
        het_loss = 0.5 * (reco / sigma2 + logsigma2)
        loss = het_loss + 0.5 * torch.log(torch.tensor(2.0) * torch.pi)
        return loss

    def plot_predictions_vs_targets_at_train(self, iteration):
        COLOR_BOOST = "firebrick"
        COLOR_SO2 = "#1E90FF"
        COLOR_SO3 = "hotpink"
        COLOR_SL4 = "#9370DB"
        COLOR_SHEAR = "darkorange"
        preds, targets, loss = self.evaluate_in_training(split="val", during_training=True)
        preds_SO2, _, loss_SO2 = self.evaluate_in_training(
            split="val", during_training=True, SO2=True
        )
        preds_SO3, _, loss_SO3 = self.evaluate_in_training(
            split="val", during_training=True, SO3=True
        )
        preds_boosted, _, loss_boosted = self.evaluate_in_training(
            split="val", during_training=True, boost=True
        )
        preds_SL4, _, loss_SL4 = self.evaluate_in_training(
            split="val", during_training=True, SL4=True
        )
        preds_shear, _, loss_shear = self.evaluate_in_training(
            split="val", during_training=True, shear=True
        )

        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        preds_SO2 = preds_SO2.cpu().numpy()
        preds_SO3 = preds_SO3.cpu().numpy()
        preds_boosted = preds_boosted.cpu().numpy()
        preds_SL4 = preds_SL4.cpu().numpy()
        preds_shear = preds_shear.cpu().numpy()

        random_uniform_number_between_0_and_1 = np.random.uniform(0, 1)
        if (
            random_uniform_number_between_0_and_1 < 0.25
        ):  # decreasing a bit the frequency of saving and plots
            tosave = {
                "preds": preds,
                "preds_boosted": preds_boosted,
                "preds_SO3": preds_SO3,
                "preds_SL4": preds_SL4,
                "preds_shear": preds_shear,
                "preds_SO2": preds_SO2,
                "targets": targets,
            }
            np.save(
                os.path.join(self.model_path, f"preds_vs_targets_{iteration}.npy"), tosave
            )

        regress_name = {
            "r": "r",
            "difference": "\Delta_{\\text{FC} - \\text{LC}}",
            "LC": "A_{\\text{LC}}",
            "FC": "A_{\\text{FC}}",
        }[self.cfg.dataset.get("regress", "r")]
        bins = np.linspace(
            *np.percentile(
                np.concatenate([preds, targets, preds_boosted]),
                [0.05, 99.95],
            ),
            64,
        )
        delta_bins = np.linspace(-5, 5, 64)
        abs_delta_bins = np.logspace(-14, 6, 64)

        # targets
        y_targets, y_targets_err = compute_hist_data(bins, targets, bayesian=False)
        y_preds, y_preds_err = compute_hist_data(bins, preds, bayesian=False)
        y_preds_boosted, y_preds_boosted_err = compute_hist_data(
            bins, preds_boosted, bayesian=False
        )
        y_preds_SO3, y_preds_SO3_err = compute_hist_data(bins, preds_SO3, bayesian=False)
        y_preds_SL4, y_preds_SL4_err = compute_hist_data(bins, preds_SL4, bayesian=False)
        y_preds_shear, y_preds_shear_err = compute_hist_data(
            bins, preds_shear, bayesian=False
        )
        y_preds_SO2, y_preds_SO2_err = compute_hist_data(bins, preds_SO2, bayesian=False)

        target_lines = [
            Line(
                y=y_targets,
                y_err=y_targets_err,
                label="Truth",
                color=TRUTH_COLOR,
            ),
            Line(
                y=y_preds,
                y_err=y_preds_err,
                y_ref=y_targets,
                label=rf"{self.name}$(x)$",
                color=NN_COLORS[self.name],
            ),
            Line(
                y=y_preds_SO2,
                y_err=y_preds_SO2_err,
                y_ref=y_targets,
                label=rf"{self.name}$(x_{{\text{{SO(2)}}}})$",
                color=COLOR_SO2,
            ),
            Line(
                y=y_preds_SO3,
                y_err=y_preds_SO3_err,
                y_ref=y_targets,
                label=rf"{self.name}$(x_{{\text{{SO(3)}}}})$",
                color=COLOR_SO3,
            ),
            Line(
                y=y_preds_boosted,
                y_err=y_preds_boosted_err,
                y_ref=y_targets,
                label=rf"{self.name}$(x_{{\text{{boost}}}})$",
                color=COLOR_BOOST,
            ),
            Line(
                y=y_preds_SL4,
                y_err=y_preds_SL4_err,
                y_ref=y_targets,
                label=rf"{self.name}$(x_{{\text{{SL(4)}}}})$",
                color=COLOR_SL4,
            ),
            Line(
                y=y_preds_shear,
                y_err=y_preds_shear_err,
                y_ref=y_targets,
                label=rf"{self.name}$(x_{{\text{{shear}}}})$",
                color=COLOR_SHEAR,
            ),
        ]

        # (rpred - rtrue) / rtrue
        delta_preds = (preds - targets) / targets
        y_delta_preds, y_delta_preds_err = compute_hist_data(
            delta_bins, delta_preds, bayesian=False
        )
        y_delta_preds_abs, y_delta_preds_abs_err = compute_hist_data(
            abs_delta_bins, np.abs(delta_preds), bayesian=False
        )

        delta_preds_boosted = (preds_boosted - targets) / targets
        y_delta_preds_boosted, y_delta_preds_boosted_err = compute_hist_data(
            delta_bins, delta_preds_boosted, bayesian=False
        )
        y_abs_delta_boosted_abs, y_abs_delta_boosted_abs_err = compute_hist_data(
            abs_delta_bins, np.abs(delta_preds_boosted), bayesian=False
        )

        delta_preds_SO3 = (preds_SO3 - targets) / targets
        y_delta_preds_SO3, y_delta_preds_SO3_err = compute_hist_data(
            delta_bins, delta_preds_SO3, bayesian=False
        )
        y_delta_preds_SO3_abs, y_delta_preds_SO3_abs_err = compute_hist_data(
            abs_delta_bins, np.abs(delta_preds_SO3), bayesian=False
        )

        delta_preds_SL4 = (preds_SL4 - targets) / targets
        y_delta_preds_SL4, y_delta_preds_SL4_err = compute_hist_data(
            delta_bins, delta_preds_SL4, bayesian=False
        )
        y_delta_preds_SL4_abs, y_delta_preds_SL4_abs_err = compute_hist_data(
            abs_delta_bins, np.abs(delta_preds_SL4), bayesian=False
        )

        delta_preds_shear = (preds_shear - targets) / targets
        y_delta_preds_shear, y_delta_preds_shear_err = compute_hist_data(
            delta_bins, delta_preds_shear, bayesian=False
        )
        y_delta_preds_shear_abs, y_delta_preds_shear_abs_err = compute_hist_data(
            abs_delta_bins, np.abs(delta_preds_shear), bayesian=False
        )

        delta_preds_SO2 = (preds_SO2 - targets) / targets
        y_delta_preds_SO2, y_delta_preds_SO2_err = compute_hist_data(
            delta_bins, delta_preds_SO2, bayesian=False
        )
        y_delta_preds_SO2_abs, y_delta_preds_SO2_abs_err = compute_hist_data(
            abs_delta_bins, np.abs(delta_preds_SO2), bayesian=False
        )

        delta_true_lines = [
            Line(
                y=y_delta_preds,
                y_err=y_delta_preds_err,
                label=rf"{self.name}$(x)$",
                color=NN_COLORS[self.name],
            ),
            Line(
                y=y_delta_preds_SO2,
                y_err=y_delta_preds_SO2_err,
                label=rf"{self.name}$(x_{{\text{{SO(2)}}}})$",
                color=COLOR_SO2,
            ),
            Line(
                y=y_delta_preds_SO3,
                y_err=y_delta_preds_SO3_err,
                label=rf"{self.name}$(x_{{\text{{SO(3)}}}})$",
                color=COLOR_SO3,
            ),
            Line(
                y=y_delta_preds_boosted,
                y_err=y_delta_preds_boosted_err,
                label=rf"{self.name}$(x_{{\text{{boost}}}})$",
                color=COLOR_BOOST,
            ),
            Line(
                y=y_delta_preds_SL4,
                y_err=y_delta_preds_SL4_err,
                label=rf"{self.name}$(x_{{\text{{SL(4)}}}})$",
                color=COLOR_SL4,
            ),
            Line(
                y=y_delta_preds_shear,
                y_err=y_delta_preds_shear_err,
                label=rf"{self.name}$(x_{{\text{{shear}}}})$",
                color=COLOR_SHEAR,
            ),
        ]
        delta_true_abs_lines = [
            Line(
                y=y_delta_preds_abs,
                y_err=y_delta_preds_abs_err,
                label=rf"{self.name}$(x)$",
                color=NN_COLORS[self.name],
            ),
            Line(
                y=y_delta_preds_SO2_abs,
                y_err=y_delta_preds_SO2_abs_err,
                label=rf"{self.name}$(x_{{\text{{SO(2)}}}})$",
                color=COLOR_SO2,
            ),
            Line(
                y=y_delta_preds_SO3_abs,
                y_err=y_delta_preds_SO3_abs_err,
                label=rf"{self.name}$(x_{{\text{{SO(3)}}}})$",
                color=COLOR_SO3,
            ),
            Line(
                y=y_abs_delta_boosted_abs,
                y_err=y_abs_delta_boosted_abs_err,
                label=rf"{self.name}$(x_{{\text{{boost}}}})$",
                color=COLOR_BOOST,
            ),
            Line(
                y=y_delta_preds_SL4_abs,
                y_err=y_delta_preds_SL4_abs_err,
                label=rf"{self.name}$(x_{{\text{{SL(4)}}}})$",
                color=COLOR_SL4,
            ),
            Line(
                y=y_delta_preds_shear_abs,
                y_err=y_delta_preds_shear_abs_err,
                label=rf"{self.name}$(x_{{\text{{shear}}}})$",
                color=COLOR_SHEAR,
            ),
        ]

        # (rpred(~x) - rpred(x)) / rpred(x)
        # # boosted
        deltas_boost = (preds_boosted - preds) / preds
        y_deltas_boost, y_deltas_boost_err = compute_hist_data(
            delta_bins, deltas_boost, bayesian=False
        )
        y_deltas_boost_abs, y_deltas_boost_abs_err = compute_hist_data(
            abs_delta_bins, np.abs(deltas_boost), bayesian=False
        )

        # # SO3
        deltas_SO3 = (preds_SO3 - preds) / preds
        y_deltas_SO3, y_deltas_SO3_err = compute_hist_data(
            delta_bins, deltas_SO3, bayesian=False
        )
        y_deltas_SO3_abs, y_deltas_SO3_abs_err = compute_hist_data(
            abs_delta_bins, np.abs(deltas_SO3), bayesian=False
        )

        # # SL4
        deltas_SL4 = (preds_SL4 - preds) / preds
        y_deltas_SL4, y_deltas_SL4_err = compute_hist_data(
            delta_bins, deltas_SL4, bayesian=False
        )
        y_deltas_SL4_abs, y_deltas_SL4_abs_err = compute_hist_data(
            abs_delta_bins, np.abs(deltas_SL4), bayesian=False
        )

        # # shear
        deltas_shear = (preds_shear - preds) / preds
        y_deltas_shear, y_deltas_shear_err = compute_hist_data(
            delta_bins, deltas_shear, bayesian=False
        )
        y_deltas_shear_abs, y_deltas_shear_abs_err = compute_hist_data(
            abs_delta_bins, np.abs(deltas_shear), bayesian=False
        )

        # # SO2
        deltas_SO2 = (preds_SO2 - preds) / preds
        y_deltas_SO2, y_deltas_SO2_err = compute_hist_data(
            delta_bins, deltas_SO2, bayesian=False
        )
        y_deltas_SO2_abs, y_deltas_SO2_abs_err = compute_hist_data(
            abs_delta_bins, np.abs(deltas_SO2), bayesian=False
        )

        mu_deltas_boost_abs = np.mean(np.abs(deltas_boost))
        mu_deltas_SO3_abs = np.mean(np.abs(deltas_SO3))
        mu_deltas_SL4_abs = np.mean(np.abs(deltas_SL4))
        mu_deltas_shear_abs = np.mean(np.abs(deltas_shear))
        mu_deltas_SO2_abs = np.mean(np.abs(deltas_SO2))

        # median
        median_deltas_boost_abs = np.median(np.abs(deltas_boost))
        median_deltas_SO3_abs = np.median(np.abs(deltas_SO3))
        median_deltas_SL4_abs = np.median(np.abs(deltas_SL4))
        median_deltas_shear_abs = np.median(np.abs(deltas_shear))
        median_deltas_SO2_abs = np.median(np.abs(deltas_SO2))

        std_deltas_boost_abs = np.std(np.abs(deltas_boost))
        std_deltas_SO3_abs = np.std(np.abs(deltas_SO3))
        std_deltas_SL4_abs = np.std(np.abs(deltas_SL4))
        std_deltas_shear_abs = np.std(np.abs(deltas_shear))
        std_deltas_SO2_abs = np.std(np.abs(deltas_SO2))

        run_dir = os.path.dirname(self.model_path)
        log_path = os.path.join(run_dir, "grokking.log")
        with open(log_path, "a") as f:
            f.write(
                f"iteration: {iteration}, "
                f"loss_median: {loss[1]}, "
                f"loss_mean: {loss[0]}, "
                f"loss_SO2_median: {loss_SO2[1]}, "
                f"loss_SO3_median: {loss_SO3[1]}, "
                f"loss_boosted_median: {loss_boosted[1]}, "
                f"loss_SL4_median: {loss_SL4[1]}, "
                f"loss_shear_median: {loss_shear[1]}, "
                f"loss_SO2_mean: {loss_SO2[0]}, "
                f"loss_SO3_mean: {loss_SO3[0]}, "
                f"loss_boosted_mean: {loss_boosted[0]}, "
                f"loss_SL4_mean: {loss_SL4[0]}, "
                f"loss_shear_mean: {loss_shear[0]}, "
                f"loss_shear_median: {loss_shear[1]}, "
                f"mu_deltas_SO2_abs: {mu_deltas_SO2_abs}, "
                f"mu_deltas_SO3_abs: {mu_deltas_SO3_abs}, "
                f"mu_deltas_SL4_abs: {mu_deltas_SL4_abs}, "
                f"mu_deltas_boost_abs: {mu_deltas_boost_abs}, "
                f"mu_deltas_shear_abs: {mu_deltas_shear_abs}, "
                f"median_deltas_SO2_abs: {median_deltas_SO2_abs}, "
                f"median_deltas_SO3_abs: {median_deltas_SO3_abs}, "
                f"median_deltas_boost_abs: {median_deltas_boost_abs}, "
                f"median_deltas_SL4_abs: {median_deltas_SL4_abs}, "
                f"median_deltas_shear_abs: {median_deltas_shear_abs}, "
                f"std_deltas_SO2_abs: {std_deltas_SO2_abs}, "
                f"std_deltas_SO3_abs: {std_deltas_SO3_abs}, "
                f"std_deltas_boost_abs: {std_deltas_boost_abs}, "
                f"std_deltas_SL4_abs: {std_deltas_SL4_abs}, "
                f"std_deltas_shear_abs: {std_deltas_shear_abs}\n"
            )

        delta_pred_lines = [
            Line(
                y=y_deltas_SO2,
                y_err=y_deltas_SO2_err,
                label=rf"{self.name}$(x_{{\text{{SO(2)}}}})$",
                color=COLOR_SO2,
            ),
            Line(
                y=y_deltas_SO3,
                y_err=y_deltas_SO3_err,
                label=rf"{self.name}$(x_{{\text{{SO(3)}}}})$",
                color=COLOR_SO3,
            ),
            Line(
                y=y_deltas_boost,
                y_err=y_deltas_boost_err,
                label=rf"{self.name}$(x_{{\text{{boost}}}})$",
                color=COLOR_BOOST,
            ),
            Line(
                y=y_deltas_SL4,
                y_err=y_deltas_SL4_err,
                label=rf"{self.name}$(x_{{\text{{SL(4)}}}})$",
                color=COLOR_SL4,
            ),
            Line(
                y=y_deltas_shear,
                y_err=y_deltas_shear_err,
                label=rf"{self.name}$(x_{{\text{{shear}}}})$",
                color=COLOR_SHEAR,
            ),
        ]
        delta_pred_abs_lines = [
            Line(
                y=y_deltas_SO2_abs,
                y_err=y_deltas_SO2_abs_err,
                label=rf"{self.name}$(x_{{\text{{SO(2)}}}})$",
                color=COLOR_SO2,
            ),
            Line(
                y=y_deltas_SO3_abs,
                y_err=y_deltas_SO3_abs_err,
                label=rf"{self.name}$(x_{{\text{{SO(3)}}}})$",
                color=COLOR_SO3,
            ),
            Line(
                y=y_deltas_boost_abs,
                y_err=y_deltas_boost_abs_err,
                label=rf"{self.name}$(x_{{\text{{boost}}}})$",
                color=COLOR_BOOST,
            ),
            Line(
                y=y_deltas_SL4_abs,
                y_err=y_deltas_SL4_abs_err,
                label=rf"{self.name}$(x_{{\text{{SL(4)}}}})$",
                color=COLOR_SL4,
            ),
            Line(
                y=y_deltas_shear_abs,
                y_err=y_deltas_shear_abs_err,
                label=rf"{self.name}$(x_{{\text{{shear}}}})$",
                color=COLOR_SHEAR,
            ),
        ]

        # Now I want to check the difference between the |delta| between 1 of the 2 Lorentz transformations and all of the other transformations

        ABSDELTA_SL4_BOOST = np.abs((deltas_SL4 - deltas_boost) / deltas_boost)
        y_absdelta_SL4_boost, y_absdelta_SL4_boost_err = compute_hist_data(
            abs_delta_bins, ABSDELTA_SL4_BOOST, bayesian=False
        )

        ABSDELTA_shear_BOOST = np.abs((deltas_shear - deltas_boost) / deltas_boost)
        y_absdelta_shear_boost, y_absdelta_shear_boost_err = compute_hist_data(
            abs_delta_bins, ABSDELTA_shear_BOOST, bayesian=False
        )

        ABSDELTA_SL4_SO3 = np.abs((deltas_SL4 - deltas_SO3) / deltas_SO3)
        y_absdelta_SL4_SO3, y_absdelta_SL4_SO3_err = compute_hist_data(
            abs_delta_bins, ABSDELTA_SL4_SO3, bayesian=False
        )

        ABSDELTA_shear_SO3 = np.abs((deltas_shear - deltas_SO3) / deltas_SO3)
        y_absdelta_shear_SO3, y_absdelta_shear_SO3_err = compute_hist_data(
            abs_delta_bins, ABSDELTA_shear_SO3, bayesian=False
        )

        ABSDELTA_SL4_SO2 = np.abs((deltas_SL4 - deltas_SO2) / deltas_SO2)
        y_absdelta_SL4_SO2, y_absdelta_SL4_SO2_err = compute_hist_data(
            abs_delta_bins, ABSDELTA_SL4_SO2, bayesian=False
        )

        ABSDELTA_shear_SO2 = np.abs((deltas_shear - deltas_SO2) / deltas_SO2)
        y_absdelta_shear_SO2, y_absdelta_shear_SO2_err = compute_hist_data(
            abs_delta_bins, ABSDELTA_shear_SO2, bayesian=False
        )

        DELTAS_LORENTZ_lines = [
            Line(
                y=y_absdelta_SL4_SO2,
                y_err=y_absdelta_SL4_SO2_err,
                label=rf"{self.name}$(x_{{\text{{SL(4)}}}})$",
                color=COLOR_SL4,
                linestyle="dotted",
            ),
            Line(
                y=y_absdelta_SL4_SO3,
                y_err=y_absdelta_SL4_SO3_err,
                label=rf"{self.name}$(x_{{\text{{SL(4)}}}})$",
                color=COLOR_SL4,
                linestyle="dashed",
            ),
            Line(
                y=y_absdelta_SL4_boost,
                y_err=y_absdelta_SL4_boost_err,
                label=rf"{self.name}$(x_{{\text{{SL(4)}}}})$",
                color=COLOR_SL4,
                linestyle="solid",
            ),
            Line(
                y=y_absdelta_shear_SO2,
                y_err=y_absdelta_shear_SO2_err,
                label=rf"{self.name}$(x_{{\text{{shear}}}})$",
                color=COLOR_SHEAR,
                linestyle="dotted",
            ),
            Line(
                y=y_absdelta_shear_SO3,
                y_err=y_absdelta_shear_SO3_err,
                label=rf"{self.name}$(x_{{\text{{shear}}}})$",
                color=COLOR_SHEAR,
                linestyle="dashed",
            ),
            Line(
                y=y_absdelta_shear_boost,
                y_err=y_absdelta_shear_boost_err,
                label=rf"{self.name}$(x_{{\text{{shear}}}})$",
                color=COLOR_SHEAR,
                linestyle="solid",
            ),
        ]

        if (
            random_uniform_number_between_0_and_1 < 0.25
        ):  # decreasing a bit the number of plots
            self.logger.info(
                f"Plotting predictions vs targets at iteration {iteration} with {len(preds)} predictions"
            )
            with PdfPages(
                os.path.join(self.model_path, f"preds_vs_targets_{iteration}.pdf")
            ) as pp:
                hist_weights_plot(
                    pp,
                    DELTAS_LORENTZ_lines,
                    abs_delta_bins,
                    show_ratios=False,
                    title=f"It. {iteration}",
                    xlabel=rf"$\left | \left ( |\Delta^{{g}}_{{\tilde{{{regress_name}}}}}|  - |\Delta^{{\text{{SO}}^{{+}}(1,3)}}_{{\tilde{{{regress_name}}}}}| \right )  / |\Delta^{{\text{{SO}}^{{+}}(1,3)}}_{{\tilde{{{regress_name}}}}}| \right |$",
                    xscale="log",
                    no_scale=True,
                    metrics=None,
                    model_name=self.name,
                )
                hist_weights_plot(
                    pp,
                    delta_pred_lines,
                    delta_bins,
                    show_ratios=False,
                    title=f"It. {iteration}",
                    xlabel=rf"$\Delta^{{g}}_{{\tilde{{{regress_name}}}}} = \frac{{\tilde{{{regress_name}}}^{{\text{{pred}}}}(x_{{g}}) - \tilde{{{regress_name}}}^{{\text{{pred}}}}(x)}}{{\tilde{{{regress_name}}}^{{\text{{pred}}}}(x)}}$",
                    xscale="linear",
                    no_scale=True,
                    metrics=None,
                    # {
                    #     "mean": DataMetric(
                    #     name="mean",
                    #     value=np.mean(deltas_boost),
                    #     unit="",
                    #     format="{:.6f}",
                    #     tex_label=r"\mu_{\text{boost}}",
                    #     ),
                    #     "std": DataMetric(
                    #         name="std",
                    #         value=np.std(deltas_boost),
                    #         unit="",
                    #         format="{:.6f}",
                    #         tex_label=r"\sigma_{\text{boost}}",
                    #     ),
                    # },
                    model_name=self.name,
                )
                hist_weights_plot(
                    pp,
                    delta_pred_abs_lines,
                    abs_delta_bins,
                    show_ratios=False,
                    title=f"It. {iteration}",
                    xlabel=rf"$|\Delta^{{g}}_{{\tilde{{{regress_name}}}}}|$",
                    xscale="log",
                    no_scale=True,
                    metrics=None,
                    # {
                    #     "mean": DataMetric(
                    #     name="mean",
                    #     value=np.mean(np.abs(deltas_boost)),
                    #     unit="",
                    #     format="{:.2e}",
                    #     tex_label=r"\mu_{\text{boost}}",
                    #     ),
                    #     "std": DataMetric(
                    #         name="std",
                    #         value=np.std(np.abs(deltas_boost)),
                    #         unit="",
                    #         format="{:.2e}",
                    #         tex_label=r"\sigma_{\text{boost}}",
                    #     ),
                    # },
                    model_name=self.name,
                    legend_kwargs={
                        "loc": "center left",
                    },
                )

                hist_weights_plot(
                    pp,
                    target_lines,
                    bins,
                    show_ratios=True,
                    title=f"It. {iteration}",
                    xlabel=rf"$\tilde{{{regress_name}}}(x)$",
                    xscale="linear",
                    no_scale=True,
                    metrics=None,
                    model_name=self.name,
                )
                hist_weights_plot(
                    pp,
                    delta_true_lines,
                    delta_bins,
                    show_ratios=False,
                    title=f"It. {iteration}",
                    xlabel=rf"$\Delta_{{\tilde{{{regress_name}}}}} = \frac{{\tilde{{{regress_name}}}^{{\text{{pred}}}} - \tilde{{{regress_name}}}^{{\text{{true}}}}}}{{\tilde{{{regress_name}}}^{{\text{{true}}}}}}$",
                    xscale="linear",
                    no_scale=True,
                    metrics=None,
                    model_name=self.name,
                )
                hist_weights_plot(
                    pp,
                    delta_true_abs_lines,
                    abs_delta_bins,
                    show_ratios=False,
                    title=f"It. {iteration}",
                    xlabel=rf"$|\Delta_{{\tilde{{{regress_name}}}}}| = \left |\frac{{\tilde{{{regress_name}}}^{{\text{{pred}}}} - \tilde{{{regress_name}}}^{{\text{{true}}}}}}{{\tilde{{{regress_name}}}^{{\text{{true}}}}}}\right |$",
                    xscale="log",
                    no_scale=True,
                    metrics=None,
                    model_name=self.name,
                )
