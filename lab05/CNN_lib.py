import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


def use_GPU():
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    device = torch.device("cuda:0" if train_on_gpu else "cpu")
    print(device)

    return device


class StandardNet(nn.Module):
    """
    This class defines a simple convolutional neural network (CNN) architecture
    for image classification.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer with 3 input channels (RGB),
                           6 output channels, and a kernel size of 5x5.
        pool (nn.MaxPool2d): Max pooling layer with a kernel size of 2x2.
        conv2 (nn.Conv2d): Second convolutional layer with 6 input channels
                           (from the first conv layer), 16 output channels,
                           and a kernel size of 5x5.
        fc1 (nn.Linear): First fully-connected layer that flattens the input
                         from the previous convolutional layers and has 120 neurons.
        fc2 (nn.Linear): Second fully-connected layer with 84 neurons.
        fc3 (nn.Linear): Output layer with 10 neurons, corresponding to the 10 classes
                         in CIFAR-10.

    Methods:
        forward(self, x): Defines the forward pass of the network.
    """

    def __init__(self):
        super(StandardNet, self).__init__()  # Call the superclass constructor
        self.conv1 = nn.Conv2d(3, 6, 5)  # First convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.conv2 = nn.Conv2d(6, 16, 5)  # Second convolutional layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # First fully-connected layer
        self.fc2 = nn.Linear(120, 84)  # Second fully-connected layer
        self.fc3 = nn.Linear(84, 10)  # Output layer

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor representing the images.

        Returns:
            torch.Tensor: Output tensor representing the class probabilities.
        """
        x = self.pool(F.relu(self.conv1(x)))  # First convolutional layer with ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Second convolutional layer with ReLU activation and pooling
        # print(x.shape)
        x = x.view(x.shape[0],-1)  # Flatten the output from convolutional layers
        # print(x.shape)
        x = F.relu(self.fc1(x))  # First fully-connected layer with ReLU activation
        x = F.relu(self.fc2(x))  # Second fully-connected layer with ReLU activation
        x = self.fc3(x)  # Output layer
        return x



def train_net(Net, trainloader,epochs,criterion,testloader = None, learning_rate = 0.001,optimizer = "SGD", momentum =0, weight_decay = 0, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), verbose = False, bestmodel_name = "last_best_model" ):

    optimizer_name = optimizer
    net = Net
    if optimizer_name == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    if optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)


    num_print_intervals = 4 # Number of times to print statistics
    num_print_intervals+=1
    print_interval = int(len(trainloader) / num_print_intervals)

    # Loop over the dataset multiple times (2 epochs in this case)
    epoch_loss = []
    test_loss = []
    acc_tot = []
    best_acc = 0
    for epoch in range(epochs):
        running_loss=[]  # Initialize a variable to track the total loss for this epoch
        epoch_running_loss=0
        # Iterate over the training data loader
        for i, data in enumerate(trainloader, 0):
            # Get the inputs (images) and labels from the current batch
            inputs, labels = data

            # Move the inputs and labels to the specified device (CPU or GPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clear the gradients accumulated in the previous iteration

            optimizer.zero_grad()

            # Training loop: forward pass, backward pass, and optimization
            # 1. Forward pass:
            outputs = net(inputs)  # Pass the input images through the network to get predictions (outputs)
            # 2. Calculate loss:
            loss = criterion(outputs, labels)  # Compute the loss based on the predictions (outputs) and ground truth labels
            # 3. Backward pass:
            loss.backward()  # Backpropagate the loss to calculate gradients for each parameter in the network
            # 4. Optimization step:
            optimizer.step()  # Update the weights and biases of the network based on the calculated gradients

            running_loss.append(loss.item())  # Accumulate the loss for this mini-batch
            if i>0 and i % print_interval == 0:  # Check batch interval
                # Print the average loss for the mini-batches
                loss_mean=np.mean(running_loss)
                if verbose :
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, i , loss_mean))

                # Reset the running loss for the next interval
                running_loss=[]
                epoch_running_loss += loss_mean

                #print("[" +"/"*i + "_"*(len(trainloader)-i) + "]")

        if not verbose and testloader is None:
            print(f"[epoch: {epoch + 1}] loss: {epoch_running_loss/(num_print_intervals-1):.4f}" )

        epoch_loss.append(epoch_running_loss/(num_print_intervals-1))

        # test loss
        correct = 0
        total = 0
        loss_test = 0
        if testloader is not None:
            net.eval()
            with torch.no_grad():
                for data in testloader:
                    image, label = data
                    image = image.to(device)
                    label = label.to(device)
                    output = net(image)

                    loss= criterion(output, label)

                    loss_test += loss

                    _, predicted = torch.max(output, 1)

                    # Update total number of test images
                    total += label.size(0)  # label.size(0) gives the batch size

                    # Count correct predictions
                    correct += (predicted == label).sum().item()  # Count true positives
            net.train()

            print(f"[epoch: {epoch + 1}] train loss: {epoch_running_loss/(num_print_intervals-1):.4f} "
                          f"test loss: {loss_test/len(testloader):.4f},test accuracy: {100 * correct / total:.1f} %")
            tlos = loss_test/len(testloader)
            test_loss.append(tlos.cpu())
            acc_tot.append(100 * correct / total)

            if 100*correct/total >= best_acc:
                torch.save(net.state_dict(), bestmodel_name + ".pth")
                best_acc = 100*correct/total

    print('Finished Training')

    if testloader is None:
        return net, epoch_loss
    else:
        return net, epoch_loss, test_loss, acc_tot