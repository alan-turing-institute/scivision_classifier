import numpy as np
from numpy.typing import ArrayLike
from classification_models.tfkeras import Classifiers
from skimage.transform import resize
import tensorflow

# type checker can't resolve tensorflow.keras in import statement
decode_predictions = tensorflow.keras.applications.imagenet_utils.decode_predictions


class ImageClassifier:
    name: str

    def __init__(self):
        model_spec, self.preprocess_input = Classifiers.get(self.name)
        self.model = model_spec(
            input_shape=(224, 224, 3),
            weights="imagenet",
            classes=1000
        )

    def predict(self, image: ArrayLike) -> str:
        """Gives the top prediction and confidence for the provided image"""
        image_arr = np.asarray(image)
        image_arr = resize(
            image_arr, (224, 224),
            preserve_range=True,
            anti_aliasing=True
        )
        image_arr = self.preprocess_input(image_arr)
        image_arr = np.expand_dims(image_arr, 0)

        y = self.model.predict(image_arr)

        _, image_class, class_confidence = decode_predictions(y, top=1)[0][0]

        return "{} : {:.2f}%".format(image_class, class_confidence * 100)


class vgg16(ImageClassifier):
    name = "vgg16"


class vgg19(ImageClassifier):
    name = "vgg19"


class resnet18(ImageClassifier):
    name = "resnet18"


class resnet34(ImageClassifier):
    name = "resnet34"


class resnet50(ImageClassifier):
    name = "resnet50"


class resnet101(ImageClassifier):
    name = "resnet101"


class resnet152(ImageClassifier):
    name = "resnet152"


class resnet50v2(ImageClassifier):
    name = "resnet50v2"


class resnet101v2(ImageClassifier):
    name = "resnet101v2"


class resnet152v2(ImageClassifier):
    name = "resnet152v2"


class resnext50(ImageClassifier):
    name = "resnext50"


class resnext101(ImageClassifier):
    name = "resnext101"


class densenet121(ImageClassifier):
    name = "densenet121"


class densenet169(ImageClassifier):
    name = "densenet169"


class densenet201(ImageClassifier):
    name = "densenet201"


class inceptionv3(ImageClassifier):
    name = "inceptionv3"


class xception(ImageClassifier):
    name = "xception"


class inceptionresnetv2(ImageClassifier):
    name = "inceptionresnetv2"


class seresnet18(ImageClassifier):
    name = "seresnet18"


class seresnet34(ImageClassifier):
    name = "seresnet34"


class seresnet50(ImageClassifier):
    name = "seresnet50"


class seresnet101(ImageClassifier):
    name = "seresnet101"


class seresnet152(ImageClassifier):
    name = "seresnet152"


class seresnext50(ImageClassifier):
    name = "seresnext50"


class seresnext101(ImageClassifier):
    name = "seresnext101"


class senet154(ImageClassifier):
    name = "senet154"


class nasnetlarge(ImageClassifier):
    name = "nasnetlarge"


class nasnetmobile(ImageClassifier):
    name = "nasnetmobile"


class mobilenet(ImageClassifier):
    name = "mobilenet"


class mobilenetv2(ImageClassifier):
    name = "mobilenetv2"
