import numpy as np
import matplotlib.pyplot as plt
import json

path = './trained_on_9718/history.json'

with open(path, 'r') as f:
    history = json.load(f)

train_loss = history['train_loss']
train_loss = [train_loss[i] / 11.0 for i in range(len(train_loss))]
val_loss = history['val_loss']
val_loss = [val_loss[i] / 1.0 for i in range(len(val_loss))]

epochs = range(0, len(train_loss)+1, 1)

plt.plot(epochs[1:], train_loss[::1], label='train loss')  # Pass epochs as x values
plt.plot(epochs[1:], val_loss[::1], label='val loss')  # Pass epochs as x values
plt.xticks(epochs[::5])
plt.yticks([])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('Loss vs Epochs')
plt.legend()
plt.show()
