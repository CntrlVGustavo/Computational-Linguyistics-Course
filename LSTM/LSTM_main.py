import LSTM_training as tr
import LSTM_data_processing as data
import LSTM_model as model
import time
import math
import torch
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker

n_iters = 300000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = tr.randomTrainingExample()
    output, loss = tr.train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = tr.categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

plt.figure()
plt.plot(all_losses)
plt.savefig('lot_l=0.02_i=300000_#1.png')


##Evaluating performance

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(data.n_categories, data.n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = model.lstm.initHidden()
    memory = model.lstm.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden, memory = model.lstm(line_tensor[i], hidden, memory)

    return output

total_count = 0
correct_guesses_count = 0
# Go through a bunch of examples and record which are correctly guessed / also calculate overall accuracy
for i in range(n_confusion):
    total_count += 1
    category, line, category_tensor, line_tensor = tr.randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = tr.categoryFromOutput(output)
    category_i = data.all_categories.index(category)
    confusion[category_i][guess_i] += 1
    if (category_i == guess_i): correct_guesses_count += 1

#print accuracy
print(str(correct_guesses_count) + " out of " + str(total_count))

# Normalize by dividing every row by its sum
for i in range(data.n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + data.all_categories, rotation=90)
ax.set_yticklabels([''] + data.all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.savefig('per_l=0.02_i=300000_#1.png')