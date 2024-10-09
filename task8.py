import time
import cv2


import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small
from sklearn.metrics import confusion_matrix


transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ToTensor()])
train_data = datasets.ImageFolder('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-5/data/train', transform = transform)



def train_model(model, trainloader, testloader, criterion, optimizer, device):
    # Saving parameters
    best_train_loss = 1e9

    # Loss lists
    train_losses = []
    test_losses = []

    results = []

    # Epoch Loop
    for epoch in range(1, 10):

        # Start timer
        t = time.time_ns()

        # Train the model
        model.train()
        train_loss = 0

        # Batch Loop
        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):

            # Move the data to the device (CPU or GPU)
            images = images.reshape(-1, 3, 64, 64).to(device)
            # labels = labels.reshape(-1, 1).to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Accumulate the loss
            train_loss = train_loss + loss.item()

        # Test the model
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        # Batch Loop
        for images, labels in tqdm(testloader, total=len(testloader), leave=False):

            # Move the data to the device (CPU or GPU)
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Accumulate the loss
            test_loss = test_loss + loss.item()

            # Get the predicted class from the maximum value in the output-list of class scores
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)

            # Accumulate the number of correct classifications
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Print the epoch statistics
        print(f'Epoch: {epoch}, Train Loss: {train_loss / len(trainloader):.4f}, Test Loss: {test_loss / len(testloader):.4f}, Test Accuracy: {correct / total:.4f}, Time: {(time.time_ns() - t) / 1e9:.2f}s')

    
        # Update loss lists
        train_losses.append(train_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        results.append((epoch, train_loss / len(trainloader), test_loss / len(testloader),correct / total))

        # Update the best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), '/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-5/best_model.pth')

        # Save the model
        torch.save(model.state_dict(), '/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-5/current_model.pth')

        # Create the confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure (figsize = (6,4))
        plt.imshow(cm, interpolation = 'nearest', cmap=plt.cm.Blues)


        plt.title('confusion matrix')
        plt.colorbar()

        threshold = cm.max()/2
        for i,j in np.ndindex(cm.shape):
            plt.text(j,i, format(cm[i,j], 'd'),
                     ha = "center", va = "center",
                     color = "white" if cm[i,j] > threshold else "black")
        plt.xlabel('predicted')
        plt.ylabel('true')            
        plt.tight_layout()
        plt.savefig('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-5/task7_confusion.png')
        plt.close()

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-5/task6_loss_plot.png')

    return results
    

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create the datasets and dataloaders
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    testloader = DataLoader(testset, batch_size=16, shuffle=False)

    loss_functions = {
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),

    }

    all_results = {}

    for loss_name, criterion in loss_functions.items():
        model = mobilenet_v3_small(weights=None, num_classes=2).to(device)

        if loss_name == 'NLLLoss':
            model.classifier[-1] = torch.nn.LogSoftmax(dim = 1)
        optimizer = optim.Adam(model.parameters(), lr=0.00005)

        results = train_model(model, trainloader, testloader, criterion, optimizer, device)
        all_results[loss_name] = results


    print("Results Summary")
    print("{:<15} {:<8} {:<12} {:<12} {:<12}".format("Loss function", "Epoch", "Train Loss", "test loss", "test acc") )
    print("=" * 60)
    for loss_name, result in all_results.items():
        for epoch, train_loss, test_loss, test_acc in result:
            print(f"{loss_name:<15} {epoch:<8} {train_loss:<12} {test_loss:<12} {test_acc:<12}")
