# NumPy Neural Network Library

This library implements a basic neural network in Python using only the NumPy library. It is designed to provide a simple, foundational understanding of neural network components and functionality.

## Features

### Components:
1. **Layers (`layer.py`):**
    - `Layer`: Represents a single layer in the neural network.
    - `LayerList`: Represents the model, which is a list of `Layer` objects.

2. **Activation Functions (`activations.py`):**
    - Includes common activation functions such as:
        - ReLU
        - Sigmoid
        - Softmax

3. **Loss Functions (`loss.py`):**
    - Provides functions to calculate the derivative of the loss function with respect to the model's output.
    - Supported loss functions:
        - Mean Squared Error (MSE)
        - Binary Cross-Entropy Loss
        - Categorical Cross-Entropy Loss

### Example Use Case
The `main.py` file demonstrates the usage of this library to build a simple handwritten digit classifier. The example uses the MNIST dataset to train and test the model.

## Prerequisites

Ensure the following Python packages are installed:
- `numpy`
- `matplotlib`
- `sklearn` (for datasets)

You can install these using pip:
```bash
pip install numpy matplotlib scikit-learn
```

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/Vineet-Vinod/NumpyNN.git
    cd NumpyNN
    ```

2. Run the main.py file:
    ```bash
    python main.py
    ```

## Additional Resources

For an in-depth walkthrough of the design and implementation, check out the [YouTube series](<insert_link_here>) detailing the concepts and build process.
