import numpy as np
import matplotlib.pyplot as plt
from src.activations import *
from src.layer import *
from src.loss import *
from sklearn import datasets
from sklearn.model_selection import train_test_split


# Load MNIST Digits Dataset
digits = datasets.load_digits()
images = digits.images
labels = digits.target


# Convert the labels into arrays of probabilities
new_labels = []
for label in labels:
    probs = [0] * 10
    probs[label] = 1
    new_labels.append(probs)

labels = np.array(new_labels)


# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, shuffle=False)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
y_train_reshaped = y_train.reshape(y_train.shape[0], -1)


# Helper function to display images and model predictions
def display_images(images, labels, title=None, predictions=None):
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(8,8))
    fig.subplots_adjust(hspace=0.8)
    if title is not None: fig.suptitle(title, fontsize=20, fontweight="bold")

    for i in range(10):
        for j in range(10):
            axs[i][j].axis("off")
            axs[i][j].imshow(images[10*i+j].reshape((8,8)), cmap="Greys")
            # Display as model prediction/actual label
            if predictions is not None: axs[i][j].set_title(f"{predictions[10*i+j]}/{labels[10*i+j]}")
            else: axs[i][j].set_title(f"A:{labels[10*i+j]}")

    plt.show()


# Helper function to convert numpy arrays of probabilities returned by the model into digit labels
def convert(classifier_predictions):
    predictions = []

    for prediction in classifier_predictions:
        curr_pred = -1
        curr_prob = 0

        for i, val in enumerate(prediction[0]):
            if val > curr_prob:
                curr_pred = i
                curr_prob = val

        predictions.append(curr_pred)

    return predictions

if __name__ == "__main__":
    # Scale input data to avoid exploding gradients
    X_train_reshaped /= 16
    X_test_reshaped /= 16

    # Hyperparameters    
    alpha = 1e-2
    batch_size = 32

    # MNIST digits classifier
    classifier = LayerList(Layer(64, 28, alpha, batch_size),
                           Layer(28, 10, alpha, batch_size,activation=Softmax()))

    # Train
    classifier.fit(X_train_reshaped, y_train_reshaped, 1000, alpha, categorical_cross_entropy_loss)

    # Evaluate
    display_images(X_test, y_test, title="NumpyNN Predictions", predictions=convert(classifier.predict(X_test_reshaped)))
