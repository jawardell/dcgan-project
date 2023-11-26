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



#batch_size = 50
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
netD = Discriminator(ngpu=0, dim_z=64, num_classes=10).to(device)


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
netD.apply(weights_init)



criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(netD.parameters(), lr=lr, weight_decay=weight_decay)

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
    "architecture": "Supervised-Baseline",
    "dataset": "STL-10",
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
    netD.train()

    
    # For each batch in the dataloader
    for i, (data, labels) in enumerate(train_loader, 0):
        # Put train data to device (CPU, GPU, or TPU)
        data_real = data.to(device)
        labels = labels.to(device)

        #  what does this do? why is this needed here?
        optimizer.zero_grad()
        
        # Forward pass batch through D
        output = netD(data_real)

        
        # Calculate loss on batch
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        


        # Compute Predicted Labels for a Batch in Training Dataset
        predicted = torch.argmax(output.data, dim=1).to(device)
	
               
 
        correct = (predicted == labels).sum().item()






        # Update All Data
        all_loss.append(loss.item())
        
        all_correct += correct
        train_running_total += labels.size(0)


        # Compute All Loss/Acc at each datapoint
        all_accuracy = all_correct / train_running_total
        all_acc.append(all_accuracy)

        print(f'iteration {i} current loss: {loss.item()} current acc: {all_accuracy}')
        
        # Log All metrics to wandb
        wandb.log({"All Loss": loss.item(), "All Accuracy": all_accuracy})




        

        # Update Epoch Data
        epoch_correct += correct
        epoch_loss_val += loss.item()
            



    # Compute Epoch Loss/Acc at end of Epoch
    epoch_accuracy = epoch_correct / num_train
    epoch_acc.append(epoch_accuracy)

    avg_epoch_loss = epoch_loss_val / len(train_loader)
    epoch_loss.append(avg_epoch_loss)

    print(f'\t\tEpoch {epoch}/{num_epochs} complete. Epoch loss {avg_epoch_loss} Epoch accuracy {epoch_accuracy}')
    
    # Log Epoch metrics to wandb
    wandb.log({"Epoch Loss": avg_epoch_loss, "Epoch Accuracy": epoch_accuracy})



    # Validation Step
    print('Starting Validation Loop...')


    
    # Refresh Validation Statistics
    print('reset Validation statistics')
    val_correct = 0
    val_loss_value = 0

    
    # Set the model to valuation mode
    netD.eval()  

    
    # Iterate over the validation dataset in batches
    with torch.no_grad():
        for data, labels in test_loader:

            # Put val data to device (CPU, GPU, or TPU)
            data_real = data.to(device)
            labels = labels.to(device)

            
            # Forward pass batch through D
            output = netD(data_real)

            # Calculate loss on validation batch
            v_loss = criterion(output, labels)
            wandb.log({"Epoch val_loss": v_loss.item()}) 

            
            # Compute Predicted Labels for a Batch in Validation Dataset
            predicted = torch.argmax(output.data, dim=1).to(device)
            val_correct += (predicted == labels).sum().item()
            
            # Update Val Data
            val_loss_value += v_loss.item()
            #val_running_total += labels.size(0)


    
    val_accuracy = val_correct / len(test_dataset)
    val_acc.append(val_accuracy)
    
    avg_val_loss = val_loss_value / len(test_loader)
    val_loss.append(avg_val_loss)
    
    print(f"\t\tValidation Epoch {epoch}, Validation Accuracy: {val_accuracy}, Validation Loss: {avg_val_loss}")


    # Log metrics to wandb
    wandb.log({"Validation Loss": avg_val_loss})
    wandb.log({"Validation Accuracy": val_accuracy})





    
    # Update best model if this epoch had the higest accuracy so far
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        print(f'best loss {best_loss}')
        best_model_state = netD.main.state_dict()



# Save the best model
if best_model_state is not None:
    PATH = '../models/supervised_baseline_{}_{}_{}.pth'.format(learning_rate, batch_size, weight_decay)
    torch.save(best_model_state, PATH)
