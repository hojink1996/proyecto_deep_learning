"""
Do the training for the model
"""
import torch

from lib.RNN import RNN

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
rnn = RNN(10, 20, dropout_prob=0.5)
rnn.to(device)

# Generate some test data
data_per_example = 5
batch_size = 20
example_size = 10
test_data = torch.randn(data_per_example, batch_size, example_size).to(device)

# Do the forward of the data
results = rnn.forward(test_data)
