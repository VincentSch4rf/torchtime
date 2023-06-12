# -*- coding: utf-8 -*-
"""
Time Series Classification with InceptionTime
=============================================

**Author**: `Vincent Scharf <vincent.scharf@smail.inf.h-brs.de>`__

In this tutorial, we are going to learn how to train an InceptionTime style classifier.

For this tutorial, we will use the Beef dataset available at the *UEA & UCR Time Series
Classification Repository* :cite:`Dau2019UCR`.
The beef dataset consists of four classes of beef spectrograms, from pure beef and beef
adulterated with varying degrees of offal.
The spectrograms are univariate time series of length 470. The train- and test set consist
of 30 sampels each.

.. note::
    ``torchtime`` provides easy access to common, publicly accessible
    datasets. Please refer to the official documentation for the list of
    available datasets.

We will do the following steps in order:

1. Load the Beef training and test datasets using ``torchtime`` and
    impute potential missing values
2. Define an InceptionTime style classifier
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

1. Load and Normalize Beef
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``torchtime``, it’s extremely easy to load datasets contained in the UCR & UEA
time series classification repository.
"""
import torch
import torch.utils.data as data
import torchtime
import torchtime.transforms as transforms

########################################################################
# The output of torchtime datasets can contain NaN values.
# We impute those missing values using ``transforms.Nan2Value()``.

########################################################################
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

transform = transforms.Compose(
    [transforms.Nan2Value()])

batch_size = 4

trainset = torchtime.datasets.UCR(root='./data', name="Beef", train=True,
                                  download=True, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchtime.datasets.UCR(root='./data', name="Beef", train=False,
                                 download=True, transform=transform)
testloader = data.DataLoader(testset, batch_size=batch_size,
                             shuffle=False, num_workers=2)

classes = ('unadulterated', 'heart', 'kidney', 'liver', 'tripe')

########################################################################
# Let us show some of the training spectrograms, for fun.

import matplotlib.pyplot as plt

def seriesshow(sequences, labels):
    """ Plot a univariate time series.
    """
    fig, ax = plt.subplots()
    for series, label in zip(sequences, labels):
        for dimension in series:
            ax.plot(dimension, label=classes[label])
    ax.legend()
    ax.grid()
    fig.show()


# get some random training sequences
dataiter = iter(trainloader)
sequences, labels = next(dataiter)

# show sequences
seriesshow(sequences, labels)
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

########################################################################
# 2. Define an InceptionTime style classifier
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We import the InceptionTime Model :cite:`IsmailFawaz2020InceptionTime` available through
# the ``torchtime.models`` package and initialize it such that it takes a 1-channel time series
# as an input and maps it onto one of the five classes defined above.

import torchtime.models as models

net = models.InceptionTime(n_inputs=1, n_classes=5)

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

########################################################################
# Let's quickly save our trained model:

PATH = './beef_classifier.pth'
torch.save(net.state_dict(), PATH)

########################################################################
# See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_
# for more details on saving PyTorch models.
#
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 10 passes over the training dataset (as it is a relatively
# small dataset with a 1:1 train/test split).
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display a series from the test set to get familiar.

dataiter = iter(testloader)
sequences, labels = next(dataiter)

# print sequences
seriesshow(sequences, labels)
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

########################################################################
# Next, let's load back in our saved model (note: saving and re-loading the model
# wasn't necessary here, we only did it to illustrate how to do so):

net = models.InceptionTime(n_inputs=1, n_classes=5)
net.load_state_dict(torch.load(PATH))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net(sequences)

########################################################################
# The outputs are energies for the 5 classes.
# The higher the energy for a class, the more the network
# thinks that the sequence is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        sequences, labels = data
        # calculate outputs by running sequences through the network
        outputs = net(sequences)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 30 test sequences: {100 * correct // total} %')

########################################################################
# That looks way better than chance, which is 20% accuracy (randomly picking
# a class out of 5 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        sequences, labels = data
        outputs = net(sequences)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

########################################################################
# Okay, so what next?
#
# How do we run these neural networks on the GPU?
#
# Training on GPU
# ^^^^^^^^^^^^^^^
# Just like how you transfer a Tensor onto the GPU, you transfer the neural
# net onto the GPU.
#
# Let's first define our device as the first visible cuda device if we have
# CUDA available:

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

########################################################################
# The rest of this section assumes that ``device`` is a CUDA device.
#
# Then these methods will recursively go over all modules and convert their
# parameters and buffers to CUDA tensors:
#
# .. code:: python
#
#     net.to(device)
#
#
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
#
# .. code:: python
#
#         inputs, labels = data[0].to(device), data[1].to(device)
#
# Why don't I notice MASSIVE speedup compared to CPU? Because your network
# is tiny.
