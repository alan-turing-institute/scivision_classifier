import numpy as np
from classification_models.tfkeras import Classifiers
from skimage.filters import gaussian
from skimage.transform import resize
from tensorflow.keras.applications.imagenet_utils import decode_predictions


def tidy_predict(self, image: np.ndarray) -> np.ndarray:
    """Gives the top prediction and confidence for the provided image"""
    image = resize(image, (224, 224), 
                   preserve_range=True, 
                   anti_aliasing=True)
    
    image = self.preprocess_input(image)
    image = np.expand_dims(image, 0)

    y = self.pretrained_model.predict(image)
    _, image_class, class_confidence = decode_predictions(y, top=1)[0][0]
    return "{} : {:.2f}%".format(image_class, class_confidence * 100)


def model_build(model_name):
    """Builds a model from the image-classifiers package"""
    model, preprocess_input = Classifiers.get(model_name)
    return model(input_shape=(224, 224, 3),
                 weights="imagenet",
                 classes=1000), preprocess_input


# model_names = ['seresnet18', 'resnet18']
# models = {}
# for model_name in model_names:
#     models[model_name] = model_build(model_name)
# 
# for model_name, model in models.items():
#     class Temp:
#         def __init__ (self):
#             self.pretrained_model = model[0]
#             self.preprocess_input = model[1]
#         def predict(self, image: np.ndarray) -> np.ndarray:
#             return tidy_predict(self, image)
#     class_name = model_name.capitalize()
#     Temp.__name__ = class_name
#     globals()[class_name] = Temp


class Resnet18:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('resnet18')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)


class Seresnet18:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('seresnet18')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)


if __name__ == "__main__":
    pass