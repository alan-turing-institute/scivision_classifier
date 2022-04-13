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



# TODO: the commented out version result in two classes named differently BUT seem to both use the same model???

# model_names = ['seresnet18', 'resnet18']
# models = {}
# for model_name in model_names:
#     models[model_name] = model_build(model_name)
# 
# for model_name in model_names:
#     class Temp:
#         def __init__ (self):
#             self.pretrained_model = models[model_name][0]
#             self.preprocess_input = models[model_name][1]
#         def predict(self, image: np.ndarray) -> np.ndarray:
#             return tidy_predict(self, image)
#     class_name = model_name.capitalize()
#     Temp.__name__ = class_name
#     globals()[class_name] = Temp
        

class vgg16:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('vgg16')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class vgg19:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('vgg19')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class resnet18:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('resnet18')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class resnet34:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('resnet34')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class resnet50:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('resnet50')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class resnet101:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('resnet101')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class resnet152:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('resnet152')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class resnet50v2:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('resnet50v2')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class resnet101v2:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('resnet101v2')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class resnet152v2:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('resnet152v2')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class resnext50:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('resnext50')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class resnext101:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('resnext101')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class densenet121:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('densenet121')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class densenet169:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('densenet169')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class densenet201:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('densenet201')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class inceptionv3:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('inceptionv3')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class xception:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('xception')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class inceptionresnetv2:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('inceptionresnetv2')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class seresnet18:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('seresnet18')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class seresnet34:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('seresnet34')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class seresnet50:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('seresnet50')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class seresnet101:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('seresnet101')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class seresnet152:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('seresnet152')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class seresnext50:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('seresnext50')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class seresnext101:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('seresnext101')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class senet154:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('senet154')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class nasnetlarge:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('nasnetlarge')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class nasnetmobile:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('nasnetmobile')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class mobilenet:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('mobilenet')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class mobilenetv2:
    def __init__(self):
        self.pretrained_model, self.preprocess_input = model_build('mobilenetv2')

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)




if __name__ == "__main__":
    pass