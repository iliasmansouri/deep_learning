from models.image_classification import alexnet, vgg16, resnet


class ModelSelector:
    @staticmethod
    def get_model(model_name):
        model_mux = {
            "alexnet": alexnet.AlexNet,
            "vgg16": vgg16.VGG16,
            "resnet": resnet.ResNet,
        }
        return model_mux.get(model_name, "Invalid model name")
