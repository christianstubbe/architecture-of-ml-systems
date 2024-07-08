# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Any


class ConvNetSimple(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.model = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 1, kernel_size=1, padding=0),
                nn.Sigmoid())
    
    def forward(self, x):
        return self.model(x)

# class 


class LitNet(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.BCELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch[:,:-1], batch[:,-1]
        outs = self.model(x.float())
        loss = self.loss(outs, y.unsqueeze(1).float())
        self.log("train_loss", value=loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[:,:-1], batch[:,-1]
        outs = self.model(x.float())
        loss = self.loss(outs, y.unsqueeze(1).float())
        values = {"val_loss": loss}
        self.log_dict(values, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        
    
    def test_step(self, batch, batch_idx):
        x, y = batch[:,:-1], batch[:,-1]
        outs = self.model(x.float())
        loss = self.loss(outs, y.unsqueeze(1).float())
        
        values = {
            "test_loss": loss,
        }
        self.log_dict(values, on_epoch=True, on_step=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self, x):
        return self.model(x)


class LitModule(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3, loss=nn.BCELoss(), optimizer:str="adamW"):
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.loss = loss
        assert optimizer in ["adamW", "adam", "RMSprop", "sgd"], "Optimizer not supported"
        self.optimizer_ = optimizer
        self.save_hyperparameters()
        

    def training_step(self, batch, batch_idx):
        x, y = batch["data"], batch["labels"]
        outs = self.model(x.float())
        loss = self.loss(outs, y.float()) # unseqeeze(1)
        self.log("train_loss", value=loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch["data"], batch["labels"]
        outs = self.model(x.float())
        loss = self.loss(outs, y.float())
        values = {"val_loss": loss}
        # self.log_dict(values, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log_dict(values, on_epoch=True, prog_bar=True, logger=True)
        
    
    def test_step(self, batch, batch_idx):
        x, y = batch["data"], batch["labels"]
        outs = self.model(x.float())
        loss = self.loss(outs, y.float())
        
        values = {
            "test_loss": loss,
        }
        self.log_dict(values, on_epoch=True, on_step=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["data"]
        return self.model(x.float())

    def configure_optimizers(self):
        if self.optimizer_ == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer_ == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_ == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer_ == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

        return optimizer
    
    def forward(self, x):
        return self.model(x)