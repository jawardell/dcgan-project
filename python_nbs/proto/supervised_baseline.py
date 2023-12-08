import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import wandb
import sys


if len(sys.argv) != 4:
    print("Usage: python supervised_baseline.py learning_rate batch_size weight_decay")
    print(sys.argv)
    sys.exit()

learning_rate = float(sys.argv[1])
batch_size = int(sys.argv[2])
weight_decay = float(sys.argv[3])
print(f'learning_rate {learning_rate}')
print(f'batch_size {batch_size}')
print(f'weight_decay {weight_decay}')


# Create a new transformation that resizes the images
transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
    # Convert the PIL Image to Torch Tensor before RandomErasing
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.2, 0.33), ratio=(0.3, 0.3), value='random'),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])



# Load STL-10 dataset
train_dataset = STL10(root='./data', split='train', transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(len(train_dataset))
print(len(train_loader))

test_dataset = STL10(root='./data', split='test', transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(len(test_dataset))
print(len(test_loader))


# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 64

# Size of feature maps in discriminator
ndf = 96

# Number of training epochs
num_epochs = 100


# Learning rate for optimizers
#lr=0.001
lr=learning_rate

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ngpu, dim_z, num_classes):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        nc = 3  # Number of input channels for the 96x96x3 image
        self.main = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 48 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 24 x 24
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 12 x 12
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 6 x 6
            nn.Conv2d(ndf * 8, dim_z, 6, 1, 0, bias=False)
        )
        self.fc = nn.Linear(dim_z, num_classes)

    def forward(self, input):
        z = self.main(input)
        z = z.view(input.size(0), -1)  # Flatten z to (batch_size, dim_z)
        c = self.fc(z)
        return c

# Instantiate the model
encoder = Discriminator(ngpu=0, dim_z=64, num_classes=10).to(device)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
encoder.apply(weights_init)



criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)

# set up wandb
wandb.login()

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="dcgan-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "batch_size": batch_size,
    "weight_decay": weight_decay,
    "epochs": num_epochs,
    "architecture": "SBL-DA",
    "dataset": "STL-10",
    }
)
    
# Training loop
best_loss = float('inf')
best_model_state = None



print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    train_loss_value = 0
    train_correct = 0
    
    # Set Network to Train Mode
    encoder.train()

    # For each batch in the dataloader
    for i, (data, labels) in enumerate(train_loader, 0): 
        data_real = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = encoder(data_real)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(output.data, dim=1).to(device)
        correct = (predicted == labels).sum().item()
        
        train_correct += correct
        train_loss_value += loss.item()
        
        train_accuracy = train_correct / len(train_dataset)
        print(f'iteration {i} current loss: {loss.item()} current acc: {train_accuracy}')


    wandb.log({"TrainAccuracy": train_accuracy})

    train_loss = train_loss_value / len(train_loader)
    wandb.log({"TrainLoss": loss.item()})


    print(f'\t\tTrain Epoch {epoch}/{num_epochs},Train Accuracy: {train_accuracy}, Train Loss: {train_loss}.')



    # Validation Step
    print('Starting Validation Loop...')
    val_correct = 0
    val_loss_value = 0
    val_running_total = 0

    # Set the model to valuation mode
    encoder.eval()


    # Iterate over the validation dataset in batches
    with torch.no_grad():
        for data, labels in test_loader:

            # Put val data to device (CPU, GPU, or TPU)
            data_real = data.to(device)
            labels = labels.to(device)


            # Forward pass batch through D
            output = encoder(data_real)

            # Calculate loss on validation batch
            v_loss = criterion(output, labels)

            # Compute Predicted Labels for a Batch in Validation Dataset
            predicted = torch.argmax(output.data, dim=1).to(device)
            val_correct += (predicted == labels).sum().item()

            # Update Val Data
            val_loss_value += v_loss.item()


    val_accuracy = val_correct / len(test_dataset)
    wandb.log({"ValidationAccuracy": val_accuracy})

    
    val_loss = val_loss_value / len(test_loader)
    wandb.log({"ValidationLoss": val_loss})

    print(f"\t\tValidation Epoch {epoch}/{num_epochs}, Validation Accuracy: {val_accuracy}, Validation Loss: {val_loss}")

    # Update best model if this epoch had the higest accuracy so far
    if train_loss < best_loss:
        best_loss = train_loss
        print(f'best loss {best_loss}')
        best_model_state = encoder.main.state_dict()




# Save the best model
if best_model_state is not None:
    PATH = '../models/sbl_da_{}_{}_{}.pth'.format(learning_rate, batch_size, weight_decay)
    torch.save(best_model_state, PATH)

