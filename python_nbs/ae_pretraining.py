import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animat
import wandb


learning_rate = float(sys.argv[1])
batch_size = int(sys.argv[2])
weight_decay = float(sys.argv[3])


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results



# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 64

# Size of feature maps in discriminator
ndf = 96

# Size of feature maps in generator
ngf = 96

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr=learning_rate

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

weight_decay = weight_decay

batch_size = batch_size
image_size = 96

# Create a new transformation that resizes the images
transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# Load STL-10 dataset
train_dataset = STL10(root='./data', split='train+unlabeled', transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(len(train_dataset))
print(len(train_loader))

test_dataset = STL10(root='./data', split='test', transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(len(test_dataset))
print(len(test_loader))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, ngpu, dim_z):
        super(Encoder, self).__init__()
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

    def forward(self, input):
        z = self.main(input)
        return z




# Instantiate the encoder
encoder = Encoder(ngpu=0, dim_z=64).to(device)

# Handle multi-GPU
if (device.type == 'cuda') and (ngpu > 1):
    encoder = nn.DataParallel(encoder, list(range(ngpu)))

# Randomly initialize all weights
encoder.apply(weights_init)

# Print the model
print(encoder)


# Generator Code

class Decoder(nn.Module):
    def __init__(self, ngpu, dim_z):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z 64x1x1, going into a convolution
            nn.ConvTranspose2d( dim_z, ngf * 8, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ndf*8) x 6 x 6
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ndf*4) x 12 x 12
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ndf*2) x 24 x 24
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ndf) x 48 x 48
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # input is (nc) x 96 x 96
        )

    def forward(self, input):
        return self.main(input)


# Instantiate the decoder
decoder = Decoder(ngpu=0, dim_z=64).to(device)

# Handle multi-GPU
if (device.type == 'cuda') and (ngpu > 1):
    decoder = nn.DataParallel(decoder, list(range(ngpu)))

# Randomly initialize all weights
decoder.apply(weights_init)

# Print the model
print(decoder)



criterion = nn.MSELoss()


params = list(encoder.parameters()) + list(decoder.parameters())

optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)



# set up wandb
wandb.login()

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="dcgan-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "weight_decay": weight_decay,
    "batch_size": batch_size,
    "architecture": "AE-Pretraining",
    "dataset": "STL-10",
    "epochs": num_epochs,
    }
)


# Training loop

best_loss = float('inf')
best_model_state = None
num_train = len(train_dataset)




# Set up total loss/acc trackers
all_loss = []
all_acc = []
all_correct = 0
train_running_total = 0



# Set up epochal loss/acc trackers
epoch_loss = []
epoch_acc = []


# Set up validation loss/acc trackers
val_loss = []
val_acc = []
val_running_total = 0



print("Starting Training Loop...")

# For each epoch
for epoch in range(num_epochs):
    # Refresh Epoch Statistics
    print('reset epoch statistics')
    epoch_correct = 0
    epoch_loss_val = 0

    
    # Set Network to Train Mode
    encoder.train()

    
    # For each batch in the dataloader
    for i, (data, _) in enumerate(train_loader, 0):

        # Put train data to device (CPU, GPU, or TPU)
        x = data.to(device)

        #  what does this do? why is this needed here?
        optimizer.zero_grad()
        
        # Forward pass batch through D
        z = encoder(x)
        
        x_bar = decoder(z)
        
        # Calculate loss on batch
        loss = criterion(x_bar, x)
        loss.backward()
        optimizer.step()
        


        # Update All Data
        all_loss.append(loss.item())
        

        print(f'iteration {i} current loss: {loss.item()}')
        
        # Log All metrics to wandb
        wandb.log({"All Loss": loss.item()})


        # Update Epoch Data
        epoch_loss_val += loss.item()
            



    # Compute Epoch Loss at end of Epoch

    avg_epoch_loss = epoch_loss_val / len(train_loader)
    epoch_loss.append(avg_epoch_loss)

    print(f'\t\tEpoch {epoch}/{num_epochs} complete. Epoch loss {avg_epoch_loss}')
    
    # Log Epoch metrics to wandb
    wandb.log({"Epoch Loss": avg_epoch_loss})



    # Validation Step
    print('Starting Validation Loop...')


    
    # Refresh Validation Statistics
    print('reset Validation statistics')
    val_correct = 0
    val_loss_value = 0

    
    # Set the model to valuation mode
    encoder.eval()  

    
    # Iterate over the validation dataset in batches
    with torch.no_grad():
        for data, _ in test_loader:
            # Put val data to device (CPU, GPU, or TPU)
            x = data.to(device)

            
            # Forward pass batch through D
            z = encoder(x)

            # Forward pass z through G
            x_hat = decoder(z)

            # Calculate loss on validation batch
            v_loss = criterion(x_hat, x)
            wandb.log({"Epoch val_loss": v_loss.item()}) 
        
            
            # Update Val Data
            val_loss_value += v_loss.item()


    val_loss_value /= len(test_loader)
    
    val_loss.append(val_loss_value)
    
    print(f"\t\tValidation Epoch {epoch}, Validation Loss: {val_loss_value}")


    # Log metrics to wandb
    wandb.log({"Validation Loss": val_loss_value})






    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        print(f'best loss {best_loss}')
        best_model_state = encoder.main.state_dict()
    
    
# Save the best model
if best_model_state is not None:
    PATH = '../models/ae_pretraining_{}_{}_{}.pth'.format(learning_rate, batch_size, weight_decay)
    torch.save(best_model_state, PATH)
