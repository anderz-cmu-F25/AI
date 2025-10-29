# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP9. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        Parameters:
        lrate (float): Learning rate for the model.
        loss_fn (callable): A loss function defined as follows:
            Parameters:
                yhat (Tensor): An (N, out_size) Tensor.
                y (Tensor): An (N,) Tensor.
            Returns:
                Tensor: A scalar Tensor that is the mean loss.
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        # For Part 1, the network should have the following architecture (in terms of hidden units):
        # in_size -> h -> out_size, where 1 <= h <= 256


        # TODO Define the network architecture (layers) based on these specifications.

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 100)  # Adjust input size based on final feature map size
        self.fc2 = nn.Linear(100, out_size)  # Output layer for 4 classes

        # Optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=lrate)
        self.activation = nn.LeakyReLU()  # Activation function
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        Parameters:
        x (Tensor): An (N, in_size) Tensor.

        Returns:
        Tensor: An (N, out_size) Tensor of output from the network.
        """
        # TODO Implement the forward pass.
        x = x.view(-1, 3, 31, 31)
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.activation(self.fc1(x))
        x = self.fc2(x)  # No activation here for logits output
        return x

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        Parameters:
        x (Tensor): An (N, in_size) Tensor representing the input data.
        y (Tensor): An (N,) Tensor representing the labels.

        Returns:
        float: The total empirical risk (mean of losses) for this batch.
        """
    
        # Zero the gradients from the previous step
        self.optimizer.zero_grad()
        
        # Forward pass: compute predicted outputs
        yhat = self.forward(x)
        
        # Compute the loss
        loss_value = self.loss_fn(yhat, y)
        
        # Backward pass: compute gradients
        loss_value.backward()
        
        # Perform a gradient descent step
        self.optimizer.step()
    
        # Important, detach and move to cpu before converting to numpy and then to python float.
        # Or just use .item() to convert to python float. It will automatically detach and move to cpu.
        return loss_value.item()



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """
    Creates and trains a NeuralNet object 'net'. Use net.step() to train the neural net
    and net(x) to evaluate the neural net.

    Parameters:
    train_set (Tensor): An (N, in_size) Tensor representing the training data.
    train_labels (Tensor): An (N,) Tensor representing the training labels.
    dev_set (Tensor): An (M,) Tensor representing the development set.
    epochs (int): The number of training epochs.
    batch_size (int, optional): The size of each training batch. Defaults to 100.

    This method must work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values if your initial choice does not work well.
    For Part 1, we recommend setting the learning rate to 0.01.

    Returns:
    list: A list of floats containing the total loss for every epoch.
        Ensure that len(losses) == epochs.
    numpy.ndarray: An (M,) NumPy array (dtype=np.int64) of estimated class labels (0,1,2, or 3) for the development set (model predictions).
    NeuralNet: A NeuralNet object.
    """
    # Standardize the training and development data
    train_set = (train_set - train_set.mean()) / train_set.std()
    dev_set = (dev_set - dev_set.mean()) / dev_set.std()

    # Initialize the neural network
    in_size = train_set.shape[1]
    out_size = 4
    net = NeuralNet(lrate=0.01, loss_fn=torch.nn.CrossEntropyLoss(), in_size=in_size, out_size=out_size)
    
    # List to store the average loss for each epoch
    losses = []
    
    # Training loop with tqdm progress bar
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = int(np.ceil(train_set.size(0) / batch_size))

        for i in range(0, train_set.size(0), batch_size):
            # Get the batch data
            x_batch = train_set[i:i+batch_size]
            y_batch = train_labels[i:i+batch_size]
            
            # Perform a training step and accumulate the loss
            batch_loss = net.step(x_batch, y_batch)
            epoch_loss += batch_loss
        
        # Calculate the average loss for the epoch and append it
        epoch_loss /= num_batches
        losses.append(epoch_loss)
    
    # Evaluate the network on the development set
    with torch.no_grad():
        predictions = net(dev_set)  # Forward pass on the dev set
        predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()  # Get the class labels as NumPy array

    # Important, don't forget to detach losses and model predictions and convert them to the right return types.
    return losses, predicted_labels, net
