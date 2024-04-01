# coding: utf-8

import sys
sys.path.insert(0, "/home/xiangzi/repository/torchvision/")

import time
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18plain, resnet34plain
import torchvision.models as models

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
batch_size = 64
learning_rate = 0.001

# Initialize transformations for data augmentation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the ImageNet Object Localization Challenge dataset
# '/hdd/home/xiangzi/ImageNetLoc/ILSVRC/Data/CLS-LOC/train'
train_dataset = torchvision.datasets.ImageFolder(
    root='/hdd/home/xiangzi/TinyImageNet/train',
    transform=transform
)
validation_dataset = torchvision.datasets.ImageFolder(
    root='/hdd/home/xiangzi/TinyImageNet/val',
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Load the ResNet model
SHORTNAME = "152identity_diagonal_down_diagonal"
model = torchvision.models.resnet152identity_diagonal_down_diagonal(pretrained=False)

# Parallelize training across multiple GPUs
#model = torch.nn.DataParallel(model)

# Set the model to run on the device
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def calc_accuracy(model, data_loader):
    correct = 0
    cnt = 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = outputs.data.cpu().topk(1, dim=1)
        preds = preds.t()
        correct += (preds == labels).float().sum()
        cnt += labels.shape[0]
    return correct / cnt

if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_index_{}_{}'.format(SHORTNAME, timestamp))
    epoch_number = 0

    EPOCHS = 50

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            train_acc = calc_accuracy(model, train_loader)
            val_acc = calc_accuracy(model, validation_loader)

        writer.add_scalars('Training vs. Validation accuracy',
                        { 'Training' : train_acc, 'Validation' : val_acc },
                        epoch_number + 1)

        writer.add_scalars('Training accuracy',
                        { 'Training' : train_acc},
                        epoch_number + 1)

        writer.add_scalars('Validation accuracy',
                        { 'Validation' : val_acc},
                        epoch_number + 1)

        writer.flush()

        # Track best performance, and save the model's state
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

        epoch_number += 1

