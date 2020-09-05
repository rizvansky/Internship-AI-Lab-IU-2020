import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomAffine, Grayscale, Resize
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import skimage.io as io
from glob import glob
from PIL.Image import fromarray


xray_data = pd.read_csv('../input/data/Data_Entry_2017.csv')

my_glob = glob('../input/data/images*/images/*.png')

full_img_paths = {os.path.basename(x): x for x in my_glob}
xray_data['full_path'] = xray_data['Image Index'].map(full_img_paths.get)

dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'] # taken from paper

for label in dummy_labels:
    xray_data[label] = xray_data['Finding Labels'].map(
        lambda result: 1.0 if label in result else 0
    )
    
xray_data['target_vector'] = xray_data.apply(lambda target: [target[dummy_labels].values], 1).map(lambda target: target[0])


class XRayDatasetFromCSV(Dataset):
    def __init__(self, csv_file, transform):
        self.data = csv_file
        self.labels = np.stack(self.data['target_vector'].values)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_name = self.data['full_path'][index]
        image = io.imread(img_name)
        image = self.transform(fromarray(image))
        target = self.labels[index]
        target = np.array([target])
        target = target.astype('float32').reshape(len(target), -1)
        target = torch.from_numpy(target)
        
        return image, target
		
		
image_size = (128, 128)

transform_train = Compose([
    Resize(image_size),
    Grayscale(),
    RandomHorizontalFlip(),
    RandomAffine(degrees=20, shear=(-0.2, 0.2, -0.2, 0.2), scale=(0.8, 1.2)),
    ToTensor()
])

transform_test = Compose([
    Resize(image_size),
    Grayscale(),
    ToTensor()
])

xray_data_train, xray_data_test = train_test_split(xray_data, test_size=0.2, shuffle=False)

dataset_train = XRayDatasetFromCSV(xray_data_train, transform_train)
dataset_test = XRayDatasetFromCSV(xray_data_test, transform_test)

train_loader = DataLoader(dataset_train, batch_size=32, num_workers=8)
test_loader = DataLoader(dataset_test, batch_size=128, num_workers=8)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2)
        )
        
        self.dense_block = nn.Sequential( 
            nn.Linear(128 * 4 * 4, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, len(dummy_labels)),
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.dense_block(x)
        x = self.sigmoid(x)
        return x
		
		
def pred_acc(original, predicted):
    return torch.round(predicted).eq(original).sum().numpy() / len(original)
	
	
def train(epoch):
    print('Epoch: {}'.format(epoch))
    net.train()
    
    running_loss = []
    running_acc = []
    
    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)           
        
        optimizer.zero_grad()
        outputs = net(images)
        
        acc_ = []
        
        for i, d in enumerate(outputs, 0):
            acc = pred_acc(targets[i].cpu(), d.cpu())
            acc_.append(acc)
            
        loss = criterion(outputs, targets)
        
        running_loss.append(loss.item())
        running_acc.append(np.asarray(acc_).mean())
        
        loss.backward()
        optimizer.step()

    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()
    
    print('Train loss is {}'.format(total_batch_loss))
    print('Accuracy is {}'.format(total_batch_acc))
    

def test(epoch):
    global best_acc
    
    print('Epoch: {}'.format(epoch))
    
    net.eval()
    
    running_loss = []
    running_acc = []
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)           
            
            outputs = net(images)
            
            acc_ = []
            
            for i, d in enumerate(outputs, 0):
                acc = pred_acc(targets[i].cpu(), d.cpu())
                acc_.append(acc)
            
            loss = criterion(outputs, targets)
            
            running_loss.append(loss.item())
            running_acc.append(np.asarray(acc_).mean())
            
    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()
            
    if total_batch_acc > best_acc:
        print('Saving..')
        best_acc = total_batch_acc
        checkpoint = {
            'model': Net(),
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
                
        torch.save(checkpoint, 'checkpoint.pth')
            
    print('Validation loss is {}'.format(total_batch_loss))
    print('Accuracy is {}'.format(total_batch_acc))
    
    
net = Net()

device = 'cpu'

if torch.cuda.is_available():
    device = 'cuda'
    
net.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())

best_acc = 0.0

for epoch in range(100):
    train(epoch)
    test(epoch)
    
print('Best reached accuracy: {}'.format(best_acc))