from torch.utils import data
from models.model_selector import ModelSelector
import pytorch_lightning as pl
from data import DataHandler


class TrainingLauncher:
    def __init__(
        self,
        path_to_data,
        model_name,
        num_classes=3,
        data_type="image_folder",
        augmentation=None,
    ):
        self.path_to_data = path_to_data
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = ModelSelector.get_model(self.model_name)
        self.data_handler = DataHandler(path_to_data, data_type, augmentation)

    def train(self):
        # TODO model get data and classes info from data_handler
        # no model.prepare data necessary
        # data_handler handles augmentation etc

        model = self.model(self.data_handler)
        model.prepare_data()
        trainer = pl.Trainer(gpus=1)
        trainer.fit(model)
