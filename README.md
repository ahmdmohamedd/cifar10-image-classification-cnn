# cifar10-image-classification-cnn
Image classification using Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 classes. This project demonstrates how to build, train, and evaluate a CNN model using TensorFlow and Keras, achieving competitive accuracy in image recognition tasks

Here's a sample `README.md` file for your project that explains the purpose, dataset, model, and how to run the code. You can modify it based on your preferences.

---

# CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)

This project demonstrates how to build a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes. The goal of the project is to train a model to classify images into one of these 10 classes.

## Dataset

CIFAR-10 is a widely used dataset for machine learning and computer vision tasks. It consists of 60,000 32x32 color images in the following 10 classes:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The dataset is already split into a training set (50,000 images) and a test set (10,000 images).

## Model Architecture

The Convolutional Neural Network (CNN) used in this project is composed of:

1. **Three convolutional layers** with ReLU activations followed by max pooling.
2. **Fully connected (Dense) layer** before the output.
3. **Output layer** with 10 neurons, one for each class, using logits for classification.

### Layers in the model:
- **Conv2D + MaxPooling2D**: Extracts features from the images.
- **Flatten**: Converts the 2D matrix to a 1D vector.
- **Dense**: Fully connected layer for classification.
- **Output layer**: Classifies the image into one of the 10 categories.

## Requirements

To run this project, you will need to install the following Python packages:

- `tensorflow`
- `numpy`
- `matplotlib`

You can install them using pip:

```bash
pip install tensorflow numpy matplotlib
```

## How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ahmdmohamedd/cifar10-image-classification-cnn/edit.git
   cd cifar10-image-classification-cnn
   ```

2. **Run the Python script**:

   ```bash
   python main.py
   ```

   The script will automatically download the CIFAR-10 dataset, build the CNN model, and start training. It will output the model's accuracy and loss during the training process.

3. **Visualize the results**:

   The script will display sample images from the dataset, plot training and validation accuracy, and display a random image with the model's predicted and true labels.

## Results

The model is trained for 10 epochs, achieving a test accuracy of around 70-80% depending on hyperparameters and training conditions. The accuracy and loss during training and validation are plotted for visualization.

## Acknowledgments

- The CIFAR-10 dataset is provided by the [Canadian Institute for Advanced Research](https://www.cs.toronto.edu/~kriz/cifar.html).
- The CNN is implemented using the TensorFlow and Keras libraries.
