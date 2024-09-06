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


    def __call__(self, output: np.array) -> np.array:
        try:
            assert(output.shape[1] == 1)
        except AssertionError:
            print(f"AssertionError assert(output.shape[1] == 1): Layer output must be an (nx1) matrix")
            sys.exit(1)


class ReLU(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()

    
    @staticmethod
    def relu(num: np.array, *args, **kwargs) -> np.array:
        num = num[0]
        if num < 0: return np.array([0])
        return np.array([num])
    

    def __call__(self, output: np.array) -> np.array:
        super().__call__(output)
        return np.apply_along_axis(self.relu, 1, output)
    

class BinaryStep(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()

    
    @staticmethod
    def b_step(num: np.array, *args, **kwargs) -> np.array:
        num = num[0]
        if num <= 0: return np.array([0])
        return np.array([1])
    

    def __call__(self, output: np.array) -> np.array:
        super().__call__(output)
        return np.apply_along_axis(self.b_step, 1, output)


class Sigmoid(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()
    

    def __call__(self, output: np.array) -> np.array:
        super().__call__(output)
        return 1 / (1 + np.exp(-output))
    

class Tanh(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()

    
    def __call__(self, output: np.array) -> np.array:
        super().__call__(output)
        return (np.exp(output) - np.exp(-output)) / (np.exp(output) + np.exp(-output))


class Softplus(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()
    

    def __call__(self, output: np.array) -> np.array:
        super().__call__(output)
        return np.log(1 + np.exp(output))
    

class ELU(ActivationFunction):
    def  __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    
    @staticmethod
    def elu(num: np.array, *args) -> np.array:
        alpha = args[0]
        num = num[0]
        if num < 0: return np.array([alpha * (exp(num) - 1)])
        return np.array([num])
    

    def __call__(self, output: np.array) -> np.array:
        super().__call__(output)
        return np.apply_along_axis(self.elu, 1, output, self.alpha)
    

class SELU(ELU):
    def  __init__(self, alpha: float, lamda: float) -> None:
        super().__init__(alpha)
        self.lamda = lamda
    

    def __call__(self, output: np.array) -> np.array:
        return self.lamda * super().__call__(output)


class PReLU(ActivationFunction):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    
    @staticmethod
    def prelu(num: np.array, *args, **kwargs) -> np.array:
        alpha = args[0]
        num = num[0]
        if num < 0: return np.array([alpha * num])
        return np.array([num])
    

    def __call__(self, output: np.array) -> np.array:
        super().__call__(output)
        return np.apply_along_axis(self.prelu, 1, output, self.alpha)


class LReLU(PReLU):
    def __init__(self) -> None:
        super().__init__(0.01)

    
    def __call__(self, output: np.array) -> np.array:
        return super().__call__(output)


class SiLU(Sigmoid):
    def __init__(self) -> None:
        super().__init__()

    
    def __call__(self, output: np.array) -> np.array:
        return super().__call__(output) * output


class Softmax(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()

    
    def __call__(self, output: np.array) -> np.array:
        super().__call__(output)
        denominator = np.sum(np.exp(output))
        return np.exp(output) / denominator
    

if __name__ == "__main__":
    relu = Softmax()
    output = np.random.rand(3,1)
    output[0] *= -1
    print(output)
    print(relu(output))
