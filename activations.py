"""
Implementing activation functions described 
at https://en.wikipedia.org/wiki/Activation_function
"""

import numpy as np
import sys
from math import exp


class ActivationFunction:
    def __init__(self) -> None:
        pass


    def __call__(self, output):
        try:
            assert(output.shape[1] == 1)
        except AssertionError:
            print(f"AssertionError assert(output.shape[1] == 1): Layer output must be an (nx1) matrix")
            sys.exit(1)

    
    def __getitem__(self, output):
        pass


class ReLU(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()

    
    def __call__(self, output):
        super().__call__(output)
        return np.maximum(output, 0)
    

    def __getitem__(self, output):
        return np.where(output < 0, 0, 1)
    

class BinaryStep(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()
    

    def __call__(self, output) :
        super().__call__(output)
        return np.where(output <= 0, 0, 1)
    

    def __getitem__(self, output):
        return np.zeros(output.shape)


class Sigmoid(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()
    

    def __call__(self, output):
        super().__call__(output)
        return 1 / (1 + np.exp(-output))
    

    def __getitem__(self, output):
        return (1 / (1 + np.exp(-output))) * (1 - (1 / (1 + np.exp(-output))))
    

class Tanh(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()

    
    def __call__(self, output):
        super().__call__(output)
        return (np.exp(output) - np.exp(-output)) / (np.exp(output) + np.exp(-output))
    

    def __getitem__(self, output):
        return 1 - ((np.exp(output) - np.exp(-output)) / (np.exp(output) + np.exp(-output)) ** 2)


class Softplus(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()
    

    def __call__(self, output):
        super().__call__(output)
        return np.log(1 + np.exp(output))
    

    def __getitem__(self, output):
        return 1 / (1 + np.exp(-output))
    

class ELU(ActivationFunction):
    def  __init__(self, alpha) -> None:
        super().__init__()
        self.alpha = alpha
    

    def __call__(self, output):
        super().__call__(output)
        return np.where(output < 0, self.alpha * (exp(output) - 1), output)
    

    def __getitem__(self, output):
        return np.where(output < 0, self.alpha * exp(output), 1)
    

class SELU(ELU):
    def  __init__(self, alpha, lamda) -> None:
        super().__init__(alpha)
        self.lamda = lamda
    

    def __call__(self, output):
        return self.lamda * super().__call__(output)
    

    def __getitem__(self, output):
        return self.lamda * super().__getitem__(output)


class PReLU(ActivationFunction):
    def __init__(self, alpha) -> None:
        super().__init__()
        self.alpha = alpha
    

    def __call__(self, output) :
        super().__call__(output)
        return np.where(output < 0, self.alpha * output, output)
    

    def __getitem__(self, output) :
        return np.where(output < 0, self.alpha, 1)


class LReLU(PReLU):
    def __init__(self) -> None:
        super().__init__(0.01)

    
    def __call__(self, output):
        return super().__call__(output)
    

    def __getitem__(self, output):
        return super().__getitem__(output)


class Softmax(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()

    
    def __call__(self, output):
        super().__call__(output)
        denominator = np.sum(np.exp(output))
        return np.exp(output) / denominator
    

    def __getitem__(self, output):
        return output * (1 - output)
    

if __name__ == "__main__":
    relu = Softmax()
    output = np.random.rand(3,1)
    output[0] *= -1
    print(output)
    print(relu(output))
