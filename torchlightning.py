import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import optuna

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=2)


class SimpleLitModel(pl.LightningModule):
    def __init__(self, learning_rate: float, hidden_size: int):
        super().__init__()
        self.save_hyperparameters() 
        self.learning_rate = learning_rate
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    dict_args = {"learning_rate": lr, "hidden_size": hidden_size}
    model = SimpleLitModel(**dict_args)
    datamodule = MNISTDataModule(batch_size=batch_size)

    wandb_logger = WandbLogger(project="hw-lightning-optuna", name=f"trial_{trial.number}")

    trainer = pl.Trainer(
        max_epochs=20, 
        logger=wandb_logger,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule)
    
    import wandb
    wandb.finish()

    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    print("Rozpoczynamy poszukiwania hiperparametrów...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("Najlepsze parametry:", study.best_params)
    print("Najlepszy wynik (Val Loss):", study.best_value)