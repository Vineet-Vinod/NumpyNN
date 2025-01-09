import numpy as np
import matplotlib.pyplot as plt
from activations import *
from layer import *
from loss import *
from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
images = digits.images
labels = digits.target

new_labels = []
for label in labels:
    probs = [0] * 10
    probs[label] = 1
    new_labels.append(probs)

labels = np.array(new_labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, shuffle=False)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
y_train_reshaped = y_train.reshape(y_train.shape[0], -1)


def display_images(images, labels, title=None, predictions=None):
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(8,8))
    fig.subplots_adjust(hspace=0.8)
    if title is not None: fig.suptitle(title, fontsize=20, fontweight="bold")

    for i in range(10):
        for j in range(10):
            axs[i][j].axis("off")
            axs[i][j].imshow(images[10*i+j].reshape((8,8)), cmap="Greys")
            if predictions is not None: axs[i][j].set_title(f"P:{predictions[10*i+j]}")
            else: axs[i][j].set_title(f"A:{labels[10*i+j]}")

    plt.show()


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
    # 2. MNIST digits classifier
    X_train_reshaped /= 16
    X_test_reshaped /= 16
    
    alpha = 1e-2
    batch_size = 32
    classifier = LayerList(Layer(64, 28, alpha, batch_size),
                           Layer(28, 10, alpha, batch_size,activation=Softmax()))

    classifier.fit(X_train_reshaped, y_train_reshaped, 1000, alpha, categorical_cross_entropy_loss)
    display_images(X_test, y_test, title="NumpyNN Predictions", predictions=convert(classifier.predict(X_test_reshaped)))

    # 1. Try to fit x^3 + y^3 + z^3
    """
    neural_net = LayerList(Layer(3, 5),
                           Layer(5, 3, bias=True, activation=ReLU()),
                           Layer(3, 1, bias=True))
    X = [[[22], [25], [24]], [[23], [19], [6]], [[13], [18], [8]], [[6], [12], [19]], [[22], [1], [7]], [[22], [7], [7]], [[19], [1], [18]], [[5], [14], [8]], [[20], [14], [13]], [[19], [10], [11]], [[17], [15], [11]], [[18], [5], [19]], [[23], [4], [6]], [[7], [15], [6]], [[23], [17], [18]], [[13], [14], [6]], [[6], [14], [22]], [[16], [18], [15]], [[22], [22], [3]], [[23], [12], [2]], [[14], [4], [10]], [[4], [3], [5]], [[7], [16], [25]], [[8], [21], [17]], [[20], [22], [20]], [[5], [9], [8]], [[21], [7], [10]], [[19], [3], [8]], [[1], [3], [13]], [[22], [24], [24]], [[17], [13], [13]], [[17], [19], [18]]]
    y = [40097, 19242, 8541, 8803, 10992, 11334, 12692, 3381, 12941, 9190, 9619, 12816, 12447, 3934, 22912, 5157, 13608, 13303, 21323, 13903, 3808, 216, 20064, 14686, 26648, 1366, 10604, 7398, 2225, 38296, 9307, 17604]
    for i in range(len(X)):
        mx = max(X[i][0][0], X[i][1][0], X[i][2][0])
        for j in range(3): X[i][j][0] /= mx
        y[i] /= mx ** 3
    
    # my = max(y)
    # y = [val / my for val in y]
    # X = [[[1],[1]],[[2],[2]],[[3],[3]],[[4],[4]],[[5],[5]],[[6],[6]]]
    # y = [2,4,6,8,10,12]
    
    training_data = np.array(X[:28])
    training_vals = np.array(y[:28])
    testing_data = np.array(X[28:])
    testing_vals = np.array(y[28:])
    
    print(MSE(neural_net.predict(testing_data), testing_vals))
    neural_net.fit(training_data, training_vals, 0.01, 50)
    print(MSE(neural_net.predict(testing_data), testing_vals))
    """