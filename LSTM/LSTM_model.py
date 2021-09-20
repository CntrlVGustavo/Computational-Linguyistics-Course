import LSTM_data_processing as data
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.memory_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden, memory):
        #passing by the forget, input, and output gate
        combined = torch.cat((input, hidden), 1)
        F = self.sigmoid(self.forget_gate(combined))
        I = self.sigmoid(self.input_gate(combined))
        output = self.sigmoid(self.output_gate(combined))

        #making the candidate memory
        M = self.tanh(self.memory_gate(combined))
        M = M * I

        #making the memory cell
        memory = memory * F
        memory = memory + M

        #making the next hidden state and the output
        hidden = output * self.tanh(memory)
        output = self.softmax(self.out(hidden))

        return output, hidden, memory

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
lstm = LSTM(data.n_letters, n_hidden, data.n_categories)
