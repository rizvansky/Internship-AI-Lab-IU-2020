import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_validation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#   it is the path that I have used to store CIFAR-10 dataset
PATH = '/content/drive/My Drive/Datasets/CIFAR10'

train_set = torchvision.datasets.CIFAR10(
    root=PATH, train=True, download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=128, shuffle=True, num_workers=2)

validation_set = torchvision.datasets.CIFAR10(
    root=PATH, train=False, download=False, transform=transform_validation)
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    #   printing the validation loss and accuracy after each epoch
        print('Training: Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
		

path = '/content/drive/My Drive/PyTorch/resnet152trained_state.pth'


def validate(epoch):
    global best_acc
    net.eval()
    validation_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validation_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            validation_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        if acc > best_acc:
            best_acc = acc
            
            #   if validation accuracy has been improved then save model's state
            torch.save(net.state_dict(), path)
            
    #   printing the validation loss and accuracy after each epoch
        print('Validation: Loss: %.3f | Acc: %.3f%% (%d/%d)'
               % (validation_loss / (batch_idx + 1), 100. * correct / total, correct, total))
	
	
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0.0

#   finetuning the model: we load a pretrained model and reset final fully connected layer
net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, len(classes))
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

#   training the network (50 epochs)
for epoch in range(50):
    train(epoch)
    validate(epoch)

best_weights = torch.load(path)
net.load_state_dict(best_weights)