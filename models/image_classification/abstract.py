from abc import abstractmethod
import pytorch_lightning as pl
from torch.optim import optimizer


class ImageClassifier(pl.LightningModule):
    def __init__(self, data_handler, loss_function, optimizer):
        super(ImageClassifier, self).__init__()
        self.num_classes = data_handler.get_num_classes()
        self.data_handler = data_handler
        self.loss_function = loss_function
        self.optimizer = optimizer

    def prepare_data(self):
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = self.data_handler.get_split()

    def shared_step(self, batch):
        x, y = batch
        y_hat = self(x)

        return self.loss_function(y_hat, y)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        tensorboard_logs = {"validation_loss": loss}
        return {"val_loss": loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        tensorboard_logs = {"validation_loss": loss}
        return {"val_loss": loss, "log": tensorboard_logs}

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=0.02)


if __name__ == "__main__":
    ic = ImageClassifier()