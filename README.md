# scivision_classifier

Model repository for the [scivision](https://scivision.readthedocs.io/) project that enables loading of pre-trained models from the [image-classifiers](https://pypi.org/project/image-classifiers/) package.

## Using the classifiers

This package has been configured for use with *scivision*, as per [these guidelines](https://scivision.readthedocs.io/en/latest/model_repository_template.html#model-repo-template).

Models can be loaded and used on data with a few lines of code, e.g.

```python
from scivision import load_pretrained_model
this_repo = 'https://github.com/alan-turing-institute/scivision_classifier'
model = load_pretrained_model(this_repo, allow_install=True, model='densenet169')
```

The full list of models that can be accepted by the `model` argument can be found on the [image-classifiers](https://pypi.org/project/image-classifiers/) package page.

You can then use the loaded model's predict function on image data loaded via *scivision* (see the [user guide](https://scivision.readthedocs.io/en/latest/user_guide.html) for details on how data is loaded via the scivision catalog):

```python
model.predict(<image data>)
```
