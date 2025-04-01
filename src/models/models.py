import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, logger, cfg, dims_in, dims_out=1):
        super().__init__()
        self.logger = logger
        self.cfg = cfg
        self.dims_in = dims_in
        self.dims_out = dims_out

        self.heteroscedastic_loss = cfg.get("heteroscedastic_loss", None)

    def init_net(self):
        layers = []
        layers.append(nn.Linear(self.dims_in, self.cfg["internal_size"]))
        layers.append(nn.ReLU())
        for _ in range(self.cfg["hidden_layers"]):
            layers.append(
                nn.Linear(self.cfg["internal_size"], self.cfg["internal_size"])
            )
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.cfg["internal_size"], self.dims_out))
        self.net = nn.Sequential(*layers)

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
        self.logger.info(f"Using channels: {channels[:self.dims_in]}")

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
            trnset, batch_size=self.cfg["batch_size"], shuffle=True
        )
        self.valloader = torch.utils.data.DataLoader(
            valset, batch_size=self.cfg["batch_size"], shuffle=True
        )
        self.tstloader = torch.utils.data.DataLoader(
            tstset,
            batch_size=self.cfg.get("batch_size_eval", self.cfg["batch_size"]),
            shuffle=False,
        )

    def init_optimizer(self):
        optim = self.cfg.get("optimizer", "adam")
        lr = float(self.cfg.get("lr", 0.001))
        wd = self.cfg.get("weight_decay", 0.0)
        if optim == "adam" or optim == "Adam":
            optimizer = torch.optim.Adam(
                self.net.parameters(), lr=lr, weight_decay=wd
            )
        else:
            raise NotImplementedError(f"Optimizer {optim} not implemented")
        self.optimizer = optimizer
        self.logger.info(f"Using optimizer {optim} with lr={lr} and weight decay={wd}")
    
    def init_scheduler(self):
        sched = self.cfg.get("scheduler", None)
        if sched == "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, len(self.trnloader) * self.cfg["nepochs"]
            )
        elif sched == "stepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=100, gamma=0.1
            )
        elif sched is None:
            scheduler = None
        else:
            raise NotImplementedError(f"Scheduler {sched} not implemented")
        self.scheduler = scheduler
        self.logger.info(f"Using scheduler {sched}")

    
    def train(self):
        self.net.train()
        nepochs = self.cfg.nepochs
        self.init_optimizer()
        self.init_scheduler()
        if self.heteroscedastic_loss is not None and self.heteroscedastic_loss.use:
            hs_loss = []
        else:
            hs_loss = None
        trn_loss = []
        val_loss = []
        trn_lr = []
        grd_norm = []
        self.logger.info(f"Training model for {nepochs} epochs")
        if self.heteroscedastic_loss.get("activate_after_its", 1000) > 1:
            self.logger.info(f"    Using heteroscedastic loss with scale {self.heteroscedastic_loss.scale} after {round(self.heteroscedastic_loss.get('activate_after_its', 1000))} iterations")
        else:
            self.logger.info(f"    Using heteroscedastic loss with scale {self.heteroscedastic_loss.scale} after {round(len(self.trnloader) * self.cfg.nepochs * self.heteroscedastic_loss.get('activate_after_its', 1000))} iterations")
        t0 = time.time()
        for epoch in range(nepochs):
            epoch_trn_losses = []
            epoch_val_losses = []
            for i, batch in enumerate(self.trnloader):
                x, weight = batch
                self.optimizer.zero_grad()
                pred = self.forward(x[:, :-1])
                target = x[:, -1:]
                current_it = epoch * len(self.trnloader) + i    
                loss, loss_terms = self.batch_loss(pred, target, weight, current_it)
                loss.backward()
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(),
                        self.cfg.get("clip_grad_norm", 10),
                        error_if_nonfinite=True,
                    )
                    .cpu()
                    .item()
                )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                trn_loss.append(loss.cpu().item())
                if "hs_loss" in loss_terms:
                    hs_loss.append(loss_terms["hs_loss"].cpu().item())
                epoch_trn_losses.append(loss.cpu().item())
                trn_lr.append(self.optimizer.param_groups[0]["lr"])
                grd_norm.append(grad_norm)

            for i, batch in enumerate(self.valloader):
                with torch.no_grad():
                    x, weight = batch
                    pred = self.forward(x[:, :-1])
                    target = x[:, -1:]
                    loss, loss_terms = self.batch_loss(pred, target, weight, current_it = self.cfg.nepochs * len(self.trnloader))
                    epoch_val_losses.append(loss.cpu().item())

            avg_trn_loss = torch.tensor(epoch_trn_losses).mean().item()
            avg_val_loss = torch.tensor(epoch_val_losses).mean().item()
            val_loss.append(avg_val_loss)
            self.losses = {
                "trn": trn_loss,
                "hs": hs_loss,
                "hs_scale": loss_terms["hs_scale"],
                "val": val_loss,
                "lr": trn_lr,
                "grad_norm": grd_norm,
            }
            if epoch == 0:
                self.logger.info(
                    f"    Epoch {epoch}: tr_loss={avg_trn_loss:.5f}, val_loss={avg_val_loss:.5f}; ETA={time.strftime('%H:%M:%S', time.gmtime((time.time() - t0) * nepochs))}"
                )
            else:
                log_every_percent = 0.25
                if epoch % max(1, int(nepochs * log_every_percent)) == 0:
                    self.logger.info(
                        f"    Epoch {epoch}: tr_loss={avg_trn_loss:.5f}, val_loss={avg_val_loss:.5f}"
                    )
        self.logger.info(f"Finished training")
    

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
        with torch.no_grad():
            t0 = time.time()
            for i, batch in enumerate(loader):
                x, weight = batch
                predicted_factors = self.predict(x[:, :-1]).detach().cpu()
                predictions.append(predicted_factors)
                t1 = time.time()
                if i == 0:
                    self.logger.info(
                        f"    Total batches: {len(loader)}. Sampling time estimate {round((t1-t0) * len(loader), 1)} seconds"
                    )
                # print info every 10 of the batche
                log_every_percent = 0.25
                if i % max(1, int(len(loader) * log_every_percent)) == 0:
                    self.logger.info(f"    Sampled batch {i+1}/{len(loader)}")
            self.logger.info(f"    Finished sampling. Saving predictions")
        predictions = torch.cat(predictions)
        return predictions
    
    def batch_loss(self, pred, target, weight, current_it, debug=False):
        if debug:
            print(pred, target)
        regression_loss = self.loss_fct(pred, target)
        
        # heteroscedastic loss
        activate_hs_loss = current_it > self.heteroscedastic_loss.get("activate_after_its", 1000) if self.heteroscedastic_loss.get("activate_after_its", 1000) > 1 else current_it/len(self.trnloader) / self.cfg.nepochs > self.heteroscedastic_loss.get("activate_after_its", 1000)
        if self.heteroscedastic_loss.use and activate_hs_loss:
            hs_loss = self.cfg.heteroscedastic_loss.get("scale", 0.001) * (target/pred).std()
        else:
            hs_loss = torch.tensor([0.]).to(pred.device)
        loss = regression_loss + hs_loss.mean()

        loss_terms = {
            "loss": loss,
            "reg_loss": regression_loss,
            "hs_loss": hs_loss,
            "hs_scale": self.cfg.heteroscedastic_loss.get("scale", 0.001),
        }
        return loss, loss_terms

class MLP(Model):
    def __init__(self, logger, process, cfg, dims_in, dims_out):
        super().__init__(logger, cfg, dims_in, dims_out)
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.loss_fct = nn.MSELoss()
        self.init_net()

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return self.forward(x)


class MDN(Model):
    def __init__(self, cfg, logger, dims_in, num_components):
        super().__init__(cfg, logger, dims_in, 3 * num_components)
        self.num_components = num_components
        self.init_net()

    def forward(self, x):
        return self.net(x)

    def batch_loss(self, pred, target, weight, debug=False):
        pi, mu, sigma = torch.split(pred, self.num_components, dim=-1)
        pi = F.softmax(pi, dim=-1)
        sigma = torch.exp(sigma)
        normal = torch.distributions.Normal(mu, sigma)

        log_prob = normal.log_prob(target.unsqueeze(-1))
        log_prob = log_prob + torch.log(pi)

        loss = -torch.logsumexp(log_prob, dim=-1).mean()

        return loss

    def predict(self, x):
        pred = self.net(x)
        pi, mu, sigma = torch.split(pred, self.num_components, dim=-1)
        pi = F.softmax(pi, dim=-1)
        sigma = torch.exp(sigma)

        component_idx = torch.multinomial(pi, 1)

        selected_mu = torch.gather(mu, dim=-1, index=component_idx)
        selected_sigma = torch.gather(sigma, dim=-1, index=component_idx)
        normal_dist = torch.distributions.Normal(selected_mu, selected_sigma)
        samples = normal_dist.sample()

        return samples.squeeze(-1)
