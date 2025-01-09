import numpy as np
import sys
from random import shuffle


class Layer:
    def __init__(self, input, output, alpha, batch_size=1, bias=False, activation=None):
        self.__input_size = input
        self.__weights = np.random.rand(input, output)
        self.__batch_size = batch_size
        self.__alpha = alpha

        self.__bias = bias
        if bias:
            self.__weights = np.vstack((self.__weights, np.ones((1, self.__weights.shape[1]))))
        
        self.__activation_layer = activation
        self.__pre_activated_output = None
        self.__curr_inputs = None
        self.__update = None


    def __call__(self, input, batch_size):
        try:
            assert(input.shape == (batch_size, self.__input_size))
        except AssertionError:
            print(f"AssertionError (input.shape == (batch_size, self.__input_size)): {input.shape} != ({batch_size}, {self.__input_size})")
            sys.exit(1)
        
        if self.__bias:
            input = np.hstack((input, np.ones((input.shape[0], 1))))

        self.__curr_inputs = np.copy(input)
        weight_prod = input @ self.__weights
        self.__pre_activated_output = np.copy(weight_prod)

        if self.__activation_layer:
            return self.__activation_layer(weight_prod)
        
        return weight_prod


    def back(self, ret):
        if self.__activation_layer:
            ret = self.__activation_layer.derivative(self.__pre_activated_output, ret)

        self.__update = (self.__curr_inputs.T) @ ret
        new_ret = ret @ (self.__weights.T)

        if self.__bias:
            return new_ret[:, :-1]
        else:
            return new_ret
    

    def update(self):
        self.__weights -= self.__alpha * self.__update
        self.__pre_activated_output = None
        self.__curr_inputs = None
        self.__update = None


    def get_batch_size(self):
        return self.__batch_size
    

    def update_alpha(self, new_alpha):
        self.__alpha = new_alpha


    def __str__(self) -> str:
        return str(self.__weights)


class LayerList:
    def __init__(self, *layers):
        if len(layers) == 0:
            self.layer_list = list()
        else:
            self.layer_list = list(layers)


    def append(self, *layers):
        for layer in layers:
            self.layer_list.append(layer)


    def __call__(self, input, batch_size):
        for layer in self.layer_list:
            input = layer(input, batch_size)
        
        return input
    

    def back(self, error):
        for layer in self.layer_list[::-1]:
            error = layer.back(error)
    

    def step(self):
        for layer in self.layer_list: layer.update()

    
    def predict(self, inputs):
        predictions = []

        for input in inputs:
            predictions.append(self(np.expand_dims(input, axis=0), 1))
        
        return predictions
    

    @staticmethod
    def batch(input_data, expected, batch_size):
        data_pts = input_data.shape[0]
        indicies = [i for i in range(data_pts)]
        shuffle(indicies)
        batched_data, batched_results = [], []

        for i in range(data_pts // batch_size):
            data_batch, expected_batch = [], []

            for j in range(batch_size):
                data_batch.append(input_data[i*batch_size+j])
                expected_batch.append(expected[i*batch_size+j])

            batched_data.append(data_batch)
            batched_results.append(expected_batch)

        return np.array(batched_data), np.array(batched_results)


    def update_alpha(self, new_alpha):
        for layer in self.layer_list:
            layer.update_alpha(new_alpha)


    def fit(self, input_data, expected, epochs, alpha, loss_deriv_func):
        if len(self.layer_list) == 0:
            return
        
        total_iter = epochs
        self.update_alpha(alpha)

        while epochs:
            epochs -= 1
            batch_size = self.layer_list[0].get_batch_size()
            batched_data, batched_expected = LayerList.batch(input_data, expected, batch_size)
            
            for idx, data_batch in enumerate(batched_data):
                output = self(data_batch, batch_size)
                self.back(loss_deriv_func(output, batched_expected[idx]))
                self.step()
            
            if epochs == total_iter // 10:
                alpha /= 10
                self.update_alpha(alpha)


    def __str__(self):
        ret = ""
        for layer in self.layer_list:
            ret += f"{layer}\n"
        return ret
