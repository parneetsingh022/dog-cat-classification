# Image Classification with Deep Learning
We will be deploying a deep learning model for image classification. This project utilizes a convolutional neural network (CNN) to differentiate between images of cats and dogs. Our dataset for this model is sourced from the widely used Cats and Dogs dataset available from https://www.microsoft.com/en-us/download/details.aspx?id=54765

Interactive demo of the model at hugging face spaces:
https://huggingface.co/spaces/parneetsingh022/cat-vs-dog

## Objective
The goal of this project is to demonstrate the practical application of a CNN in image classification. We aim to achieve the following:

- Preprocess and prepare a dataset for model training.
- Construct and train a CNN on this dataset.
- Evaluate the model's performance to ensure it meets our criteria for accuracy and efficiency.

## Dataset
We use the Cats and Dogs dataset, which provides a balanced number of images for both classes, making it ideal for our classification model. This dataset is a common benchmark in the field of computer vision and helps in understanding model behavior on real-world data.

## Tools and Libraries
In this project, we will utilize the following tools and libraries:

- PyTorch: Our main framework for modeling and training.
- NumPy: For handling data manipulations.
- Matplotlib: To visualize images and training results.

# Loading and using the model.
In order to download and use the model follow these steps:
1. Clone this reposetory:
```
git clone https://github.com/parneetsingh022/dog-cat-classification.git
```

2. Download transformers liberary
```
pip install transformers
```
3.
After installation, you can utilize the model in other scripts outside of this directory `custom_classifier` as described below:

Required Imports:
```python
import torch
from PIL import Image
from torchvision import transforms

from catdog_classifier.configuration import CustomModelConfig
from catdog_classifier.model import CustomClassifier
```

Loading the model:
```python
model_name = "parneetsingh022/dog-cat-classification"
config = CustomModelConfig.from_pretrained(model_name)
model = CustomClassifier.from_pretrained(model_name, config=config)
```

Pridicting probability of individual class:
```python
# Load an example image
image_path = "dog.jpeg"
outputs = model.predict(image_path)
print(outputs)
```
Output: {'cat': 0.003, 'dog': 0.997}

Getting class name instead of probabilities:
```python
# Load an example image
image_path = "dog_new.jpeg"
outputs = model.predict(image_path, get_class=True)

print(output)
```
Output: 'dog'
