import numpy as np

class Linear():
    def __init__(self, input_size, output_size, hidden_size, features):
        self.input_size = input_size
        self.output_size = output_size
        
        self.hidden_weights = np.random.random((features, hidden_size))
        self.hidden_bias = np.random.random((1, hidden_size))

        self.output_weights = np.random.random((hidden_size, output_size))
        self.output_bias = np.random.random((1, output_size))
    
    def forward(self, input, hidden):
        return
    
    def backward():
        return
