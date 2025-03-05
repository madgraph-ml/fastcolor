import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import time


class Model(nn.Module):
    def __init__(self, params, logger, dims_in, dims_out = 1):
        super().__init__()
        self.params = params
        self.logger = logger
        self.dims_in = dims_in
        self.dims_out = dims_out

    def init_network(self):
        layers = []
        layers.append(nn.Linear(self.dims_in, self.params["internal_size"]))
        layers.append(nn.ReLU())
        for _ in range(self.params["hidden_layers"]):
            layers.append(
                nn.Linear(self.params["internal_size"], self.params["internal_size"])
            )
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.params["internal_size"], self.dims_out))
        self.network = nn.Sequential(*layers)
    
    
    
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
        self.logger.info(f"Using channels: {channels}")

        # apply stuff to make it such that only the channels of the data are used
        if weights is None:
            weights = {
                f"{split}": torch.ones(data_ppd[f"{split}"].shape[0])
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
            trnset, batch_size=self.params["batch_size"], shuffle=True
        )
        self.valloader = torch.utils.data.DataLoader(
            valset, batch_size=self.params["batch_size"], shuffle=True
        )
        self.tstloader = torch.utils.data.DataLoader(
            tstset,
            batch_size=self.params.get("batch_size_eval", self.params["batch_size"]),
            shuffle=False,
        )

    def train(self):
        self.network.train()
        n_epochs = self.params["n_epochs"]
        lr = self.params["lr"]
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, len(self.trnloader) * n_epochs
        )
        self.logger.info(f"Training model for {n_epochs} epochs with lr={lr}")
        t0 = time.time()
        trn_loss = []
        val_loss = []
        trn_lr = []
        grd_norm = []
        for epoch in range(n_epochs):
            epoch_trn_losses = []
            epoch_val_losses = []
            for i, batch in enumerate(self.trnloader):
                x, weight = batch
                optimizer.zero_grad()

                pred = self.forward(x[:, :-1])
                target = x[:, -1:]

                loss = self.batch_loss(pred, target, weight)
                loss.backward()
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        2,
                        error_if_nonfinite=True,
                    )
                    .cpu()
                    .item()
                )
                optimizer.step()
                scheduler.step()
                trn_loss.append(loss.cpu().item())
                epoch_trn_losses.append(loss.cpu().item())
                trn_lr.append(optimizer.param_groups[0]["lr"])
                grd_norm.append(grad_norm)

            for i, batch in enumerate(self.valloader):
                with torch.no_grad():
                    x, weight = batch
                    pred = self.forward(x[:, :-1])
                    target = x[:, -1:]
                    loss = self.batch_loss(pred, target, weight)
                    epoch_val_losses.append(loss.cpu().item())

            avg_trn_loss = torch.tensor(epoch_trn_losses).mean().item()
            avg_val_loss = torch.tensor(epoch_val_losses).mean().item()
            val_loss.append(avg_val_loss)
            self.losses = {
                "trn": trn_loss,
                "val": val_loss,
                "lr": trn_lr,
                "grad_norm": grd_norm,
            }
            if epoch == 0:
                self.logger.info(
                    f"    Epoch {epoch}: tr_loss={avg_trn_loss:.5f}, val_loss={avg_val_loss:.5f}; ETA={time.strftime('%H:%M:%S', time.gmtime((time.time() - t0) * n_epochs))}"
                )
            else:
                log_every_percent = 0.25
                if epoch % int(n_epochs * log_every_percent) == 0:
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
                if i % int(len(loader) * log_every_percent) == 0:
                    self.logger.info(f"    Sampled batch {i+1}/{len(loader)}")
            self.logger.info(f"    Finished sampling. Saving predictions")
        predictions = torch.cat(predictions)
        return predictions


class MLP(Model):
    def __init__(self, params, logger, dims_in, dims_out):
        super().__init__(params, logger, dims_in, dims_out)
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.loss = nn.MSELoss(reduction="none")
        self.init_network()

    def forward(self, x):
        return self.network(x)
    
    def predict(self, x):
        return self.forward(x)

    def batch_loss(self, pred, target, weight, debug=False):
        if debug:
            print(pred, target)
        loss = self.loss(pred, target)
        loss = (loss * weight).sum() / weight.sum() 
        return loss
    
    
class MDN(Model):
    def __init__(self, params, logger, dims_in, num_components):
        super().__init__(params, logger, dims_in, 3*num_components)
        self.num_components = num_components
        self.init_network()
    
    def forward(self, x):
        return self.network(x)
    
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
        pred = self.network(x)
        pi, mu, sigma = torch.split(pred, self.num_components, dim=-1)
        pi = F.softmax(pi, dim=-1)
        sigma = torch.exp(sigma)
        
        component_idx = torch.multinomial(pi, 1)

        selected_mu = torch.gather(mu, dim=-1, index=component_idx)
        selected_sigma = torch.gather(sigma, dim=-1, index=component_idx)
        normal_dist = torch.distributions.Normal(selected_mu, selected_sigma)
        samples = normal_dist.sample()

        return samples.squeeze(-1)