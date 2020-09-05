import pandas as pd
import os
import torchvision
import torchvision.transforms as transform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils_xray import *


#	dataset \in ['chest_xray_pneumonia', 'rsna', 'tb', 'gb7', 'chest14', 'all']
dataset = 'chest_xray_pneumonia'

train_csv = pd.read_csv(os.path.join(dataset, 'train.csv'))
val_csv = pd.read_csv(os.path.join(dataset, 'val.csv'))
test_csv = pd.read_csv(os.path.join(dataset, 'test.csv'))

mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]

train_transforms = transform.Compose([
	transform.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.9, 1.1)),
	transform.RandomHorizontalFlip(),
	transform.Resize((224, 224)),
	transform.CenterCrop(224), 
	transform.ToTensor(),
	transform.Normalize(mean, std)
])

val_transforms = transform.Compose([
	transform.Resize((224, 224)),
	transform.ToTensor(),
	transform.Normalize(mean, std)
])

test_transforms = val_transforms

train_set = None
val_set = None
test_set = None

if dataset == 'tb':
	train_set = XRayDatasetFromCSV(train_csv, train_transforms, invert=True)
	val_set = XRayDatasetFromCSV(val_csv, val_transforms, invert=True)
	test_set = XRayDatasetFromCSV(test_csv, test_transforms, invert=True)
else:
	train_set = XRayDatasetFromCSV(train_csv, train_transforms)
	val_set = XRayDatasetFromCSV(val_csv, val_transforms)
	test_set = XRayDatasetFromCSV(test_csv, test_transforms)
	
device = 'cpu'
if torch.cuda.is_available():
	device = 'cuda'

train_loader = DataLoader(train_set, batch_size=20, shuffle=True)
val_loader = DataLoader(val_set, batch_size=20, shuffle=False)
test_loader = DataLoader(test_set, batch_size=20, shuffle=False)

net = torchvision.models.resnet50(pretrained=True)
num_ftrs = net.fc.in_features

for param in net.parameters():
	param.requires_grad = False

net.fc = nn.Linear(num_ftrs, 2)    

net = net.to(device)

#summary(net, (3, 224, 224))

optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)

#	can be nn.CrossEntropyLoss(weight=calculate_weights(distribution, device))
criterion = nn.CrossEntropyLoss()

path = os.path.join('/home/intern/r.iskaliev/work', dataset, 'net.pth')

aux_metrics_train = {
	'losses' : [],
	'accuracies' : [],
	'predictions' : [],
	'ground_truth' : []
}

aux_metrics_val = {
	'losses' : [],
	'accuracies' : [],
	'predictionsl' : [],
	'ground_truth' : [],
	'best_val_auc' : 0.0
}

metrics_train = []
metrics_val = []

num_epochs = 50

outer = tqdm.tqdm(total=num_epochs, desc=dataset + ', epochs', position=0)
best_val_auc_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}')
last_saved_model_epoch_log = tqdm.tqdm(total=0, position=9, bar_format='{desc}')

for epoch in range(num_epochs):
	aux_metrics_train['predictions'] = []
	aux_metrics_train['ground_truth'] = []
	train(net, device, train_loader, optimizer, criterion, aux_metrics_train)
	metrics_train.append(calculate_metrics(aux_metrics_train, 'Train'))

	aux_metrics_val['predictions'] = []
	aux_metrics_val['ground_truth'] = []
	prev_auc = aux_metrics_val['best_val_auc']
	validate(net, device, val_loader, criterion, path, aux_metrics_val)

	if prev_auc < aux_metrics_val['best_val_auc']:
		last_saved_model_epoch_log.set_description_str('Saving was at: {} epoch'.format(epoch + 1))

	best_val_auc_log.set_description_str('Best AUC score on validation: {:.4f}'.format(aux_metrics_val['best_val_auc']))
	metrics_val.append(calculate_metrics(aux_metrics_val, 'Validation'))

	show_metrics(metrics_train[len(metrics_train) - 1])
	show_metrics(metrics_val[len(metrics_val) - 1])

	metrics_training = {
		'train': metrics_train,
		'val': metrics_val,
	}   

	torch.save(metrics_training, os.path.join('/home/intern/r.iskaliev/work', dataset, 'train_metrics.pth'))

	outer.update(1)