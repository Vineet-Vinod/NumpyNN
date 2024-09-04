import numpy as np


class Layer:
    def __init__(self, input: int, output: int, bias:bool=True) -> None:
        self.__input_size = input
        self.__output_size = output
        self.__weights = np.random.rand(output, input)
        self.__bias = np.random.rand(output, 1) if bias else None

    def forward(self, input: np.array):
        assert(input.shape == (self.__input_size, 1))
        weight_prod = self.__weights @ input
        
        if self.__bias is not None:
            weight_prod += self.__bias
        return weight_prod
    
    def print_wb(self):
        print(self.__weights)
        if self.__bias is not None:
            print(self.__bias)


if __name__ == "__main__":
    single_layer = Layer(3, 5)
    input = np.random.rand(3, 1)
    print(input)
    single_layer.print_wb()
    print(single_layer.forward(input))
"""
    O
O   O
O   O
O   O
    O

w11x1 + w12x2 + w13x3 + b1
w21x1 + w22x2 + w22x3 + b2
w31x1 + w32x2 + w33x3 + b1
w41x1 + w42x2 + w43x3 + b2
w51x1 + w52x2 + w53x3 + b1

(5*3) @ (3*1)
"""