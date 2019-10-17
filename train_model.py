"""
Do the training for the model
"""
import torch

from lib.RNN import RNN
from torch import optim, nn

"""
Parameters
"""
# Parameters for the network
hidden_layer_neurons = 20

# Parameters for training data
data_per_example = 5
batch_size = 20
example_size = 10

# Parameters for learning
lr = 0.01
dropout_probability = 0.5
n_epochs = 100

"""
Predictions and training
"""
# Check if CUDA is available
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


# Define the RNN
rnn = RNN(example_size, hidden_layer_neurons, dropout_prob=dropout_probability)
rnn.to(device)

# Generate some test data
test_data = torch.randn(data_per_example, batch_size, example_size).to(device)

# Do the forward of the data
results = rnn.forward(test_data)

# Loss function and optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=lr)


training_data = []
# Train
for epoch in range(n_epochs):
    rnn.train()

    for index in training_data:
        # Make gradients zero
        optimizer.zero_grad()

        # Forward
        outputs = rnn.forward(test_data)

        # Value for loss
        # loss_value = loss(outputs, tags)
        # loss_value.backward()
        optimizer.step()