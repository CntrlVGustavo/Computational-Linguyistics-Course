from torch import nn
import LSTM_data_processing as data
import LSTM_model as model
import random
import torch
import torch.nn as nn

## Helper Functions

#interprets the output of the NN
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return data.all_categories[category_i], category_i

#funtions to get a random training example
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(data.all_categories)
    line = randomChoice(data.category_lines[category])
    category_tensor = torch.tensor([data.all_categories.index(category)], dtype=torch.long)
    line_tensor = data.lineToTensor(line)
    return category, line, category_tensor, line_tensor

#loss function
criterion = nn.NLLLoss()

#learning rate
learning_rate = 0.01 

#traiuning function
def train(category_tensor, line_tensor):
    hidden = model.lstm.initHidden()
    memory = model.lstm.initHidden()

    model.lstm.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden, memory = model.lstm(line_tensor[i], hidden, memory)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in model.lstm.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()