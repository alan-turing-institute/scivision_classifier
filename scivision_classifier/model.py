import numpy as np
from classification_models.tfkeras import Classifiers
from skimage.filters import gaussian
from skimage.transform import resize
from tensorflow.keras.applications.imagenet_utils import decode_predictions


class ImageClassifier:
    def __init__(self, model_name: str = "resnet18"):
        model, self.preprocess_input = Classifiers.get(model_name)

        # can build the model here, so we can reuse the prediction function
        self.pretrained_model = model(
            input_shape=(224, 224, 3), weights="imagenet", classes=1000
        )

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Gives the top prediction and confidence for the provided image"""
        image = resize(image, (224, 224), 
                       preserve_range=True, 
                       anti_aliasing=True)
        
        image = self.preprocess_input(image)
        image = np.expand_dims(image, 0)

        y = self.pretrained_model.predict(image)
        _, image_class, class_confidence = decode_predictions(y, top=1)[0][0]
        return "{} : {:.2f}%".format(image_class, class_confidence * 100)


if __name__ == "__main__":
    pass