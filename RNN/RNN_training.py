from torch import nn
import RNN_data_processing as data
import RNN_model as model
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

#defining the loss function
criterion = nn.NLLLoss()

learning_rate = 0.03 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = model.rnn.initHidden()

    model.rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model.rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in model.rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()