import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim


class simplecnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(16*14*14,10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        return x 
    

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = Dataloader(train_dataset, batch_size=64, shuffle=True)
test_loader = Dataloader(test_dataset, batch_size=64, shuffle=False)


model = simplecnn()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader):

    model.train()

    total_loss = 0

    for image, labels in loader:

        optimizer.zero_grad()
        output = model(image)

        loss=criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(total_loss/len(loader))


def evaluate(model, loader):

    model.eval()

    correct=0
    total=0

    with torch.no_grad():
        for image, label in loader:

            output = model(image)
            _, predicted = torch.max(output, 1)

            total += label.size(0)
            correct += (predicted==label).sum().item()

    accuracy = 100*correct / total


for epoch in range(5):
    print(f"\n Epoch{epoch+1}")
    train(model, train_loader)
    evaluate(model, test_loader)