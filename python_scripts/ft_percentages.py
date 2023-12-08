import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import sys
import random



learning_rate = 0.0001
batch_size = 256
weight_decay = 0.04


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

num_train = len(train_dataset)
nc = 3
ndf = 96
num_epochs = 100
lr=learning_rate
beta1 = 0.5
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class Encoder(nn.Module):
    def __init__(self, ngpu, dim_z, num_classes):
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
        self.fc = nn.Linear(dim_z, num_classes)

    def forward(self, input):
        z = self.main(input)
        z = z.view(input.size(0), -1)  # Flatten z to (batch_size, dim_z)
        c = self.fc(z)
        return c

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Use Kaiming initialization for Conv layers
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def generate_n_samples_per_class(dataset, labels, num_samples_per_class):
    # generate fake class distribution
    print(f'percent {percent}')
    num_samples = len(dataset)
    num_classes = len(np.unique(labels))
    indexes = range(len(labels))
    #print (np.histogram(labels, bins=range(num_classes+1)))

    # parameters
    unique_classes = set(labels)
    selected_indexes = []
    #num_samples_per_class = 10

    # random sampling
    random_indexes = list(range(len(labels)))
    random.shuffle(random_indexes)

    # counters and flag to stop
    counters = {cls: [] for cls in unique_classes}
    is_full = [False for _ in range(len(unique_classes))]

    # sample until class full-filled
    for idx in random_indexes:
        cls = labels[idx]
        if len(counters[cls]) < num_samples_per_class:
            counters[cls].append(idx)
            if len(counters[cls]) == num_samples_per_class:
                is_full[cls] = True
        if all(is_full):
            break
    #print (counters)

    # combine all indexes by class from the dictionary
    all_indexes = []
    for k in counters.keys():
        all_indexes += counters[k]
    #print (all_indexes)

    # check final results
    all_classes_selected = []
    for k in counters.keys():
        for idx in counters[k]:
            all_classes_selected.append(labels[idx])
    #print (np.histogram(all_classes_selected, bins=range(num_classes+1)))

    return all_indexes




# Lists to store data for plotting
percentages = []
avg_val_accs = []



# Training loop for different percentages of labeled data
for percent in range(10, 101, 10):  # Train on 10%, 20%, ..., 100% of the labeled data
    # Calculate the number of samples to use
    #num_samples = int(num_train * percent / 100)
    
    labels = train_dataset.labels
    num_classes = len(np.unique(labels))
    samples_per_class = (len(train_dataset) * percent) / num_classes
    all_indexes = generate_n_samples_per_class(train_dataset, labels, samples_per_class)

    
    # Create a subset of the dataset with the desired percentage
    subset_train_dataset = torch.utils.data.Subset(train_dataset, indices=all_indexes)
    subset_train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)


    model = Encoder(ngpu=1, dim_z=64, num_classes=10).to(device)
    PATH='/data/users2/jwardell1/dcgan-project/models/ae_pretraining_0.0001_256_0.0004.pth'
    model.main.load_state_dict(torch.load(PATH))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    
        
    # Training loop
    best_loss = float('inf')
    best_model_state = None
    
    
    
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        train_loss_value = 0
        train_correct = 0
        
        # Set Network to Train Mode
        model.train()
    
        # For each batch in the dataloader
        for i, (data, labels) in enumerate(train_loader, 0): 
            data_real = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(data_real)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    
            predicted = torch.argmax(output.data, dim=1).to(device)
            correct = (predicted == labels).sum().item()
            
            train_correct += correct
            train_loss_value += loss.item()
            
            train_accuracy = train_correct / len(train_dataset)
            print(f'iteration {i} current loss: {loss.item()} current acc: {train_accuracy}')
    
    
    
        train_loss = train_loss_value / len(train_loader)
    
    
        print(f'\t\tTrain Epoch {epoch}/{num_epochs},Train Accuracy: {train_accuracy}, Train Loss: {train_loss}.')
    
    
    
        # Validation Step
        print('Starting Validation Loop...')
        val_correct = 0
        val_loss_value = 0
        val_running_total = 0
        val_acc = []
    
        # Set the model to valuation mode
        model.eval()
    
    
        # Iterate over the validation dataset in batches
        with torch.no_grad():
            for data, labels in test_loader:
    
                # Put val data to device (CPU, GPU, or TPU)
                data_real = data.to(device)
                labels = labels.to(device)
    
    
                # Forward pass batch through D
                output = model(data_real)
    
                # Calculate loss on validation batch
                v_loss = criterion(output, labels)
    
                # Compute Predicted Labels for a Batch in Validation Dataset
                predicted = torch.argmax(output.data, dim=1).to(device)
                val_correct += (predicted == labels).sum().item()
    
                # Update Val Data
                val_loss_value += v_loss.item()
    
    
        val_accuracy = val_correct / len(test_dataset)
        val_acc.append(val_accuracy)
    
        
        val_loss = val_loss_value / len(test_loader)
    
        print(f"\t\tValidation Epoch {epoch}/{num_epochs}, Validation Accuracy: {val_accuracy}, Validation Loss: {val_loss}")
    
        # Update best model if this epoch had the higest accuracy so far
        if train_loss < best_loss:
            best_loss = train_loss
            print(f'best loss {best_loss}')
            best_model_state = model.main.state_dict()
    

    # Record avg val acc across all epochs for each percent
    percentages.append(percent)
    avg_val_accs.append(sum(val_acc) / len(val_acc))
    print(f'percent {percent}')
    print(f'avg val acc {sum(val_acc) / len(val_acc)}')



# Save the best model
if best_model_state is not None:
    PATH = '../models/sbl_da_ki_pct_{}_{}_{}.pth'.format(learning_rate, batch_size, weight_decay)
    torch.save(best_model_state, PATH)


plt.plot(percentages, avg_val_accs, marker='o')
plt.xlabel('Percentage of Labeled Data')
plt.ylabel('Validation Accuracy')
plt.title('FT Validation Accuracy vs Percentage of Labeled Data')
plt.grid(True)
plt.savefig('../models/sbl_acc_per.png')
np.save('../models/ft_avg_val_accs_{}_{}_{}.npy'.format(learning_rate, batch_size, weight_decay), avg_val_accs)
