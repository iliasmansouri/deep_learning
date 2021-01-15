from models.model_selector import ModelSelector
import pytorch_lightning as pl


class TrainingLauncher:
    def __init__(self, path_to_data, model_name, num_classes=3):
        self.path_to_data = path_to_data
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = ModelSelector.get_model(self.model_name)

    def train(self):
        model = self.model(self.path_to_data, self.num_classes)
        model.prepare_data()
        trainer = pl.Trainer(gpus=1)
        trainer.fit(model)
