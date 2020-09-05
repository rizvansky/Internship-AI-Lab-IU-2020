import numpy as np
import pydicom
import torchvision
import torchvision.transforms as transform
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import tqdm.notebook as tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut


#	class that generates dataset from .csv file
class XRayDatasetFromCSV(Dataset):
    def __init__(self, csv_file, transforms, invert=None):
        self.data = csv_file
        self.transforms = transforms
        self.invert = invert

    def __len__(self):
        return len(self.data)

    def read_dicom_image(self, img_name, invert):
        dcm = pydicom.read_file(img_name)
        img = apply_voi_lut(apply_modality_lut(dcm.pixel_array, dcm), dcm)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        if invert != None:
            img = np.invert(img)
        img = np.expand_dims(img, axis=-1)
        return img

    def __getitem__(self, i):
        img_name = self.data['Path to img'][i]

        img = None
        if '.jpeg' in img_name or '.png' in img_name:
            img = Image.open(img_name).convert('RGB')
            img = self.transforms(img)
        else:
            img = self.read_dicom_image(img_name, self.invert)
            img = transform.ToPILImage()(img).convert('RGB')
            img = self.transforms(img)

        target = self.data['Target'][i]

        return img, target


#	custom Sampler class
class ImbalancedDataSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, label_counts, indices=None, num_samples=None):
        self.indices = list(range(len(dataset))) if indices == None else indices
        self.num_samples = len(dataset) if num_samples==None else num_samples 
        
        # weight for each sample
        weights = [1.0 / label_counts[dataset[idx][1]] for idx in self.indices]
        self.weights = torch.tensor(weights, dtype=torch.float32)
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


#	function that calculates some metrics
def calculate_metrics(aux_metrics, stage):
    ground_truth = aux_metrics['ground_truth']
    predictions = aux_metrics['predictions']

    cf_matrix = [[0, 0], [0, 0]]

    for i in range(len(ground_truth)):
        if ground_truth[i] == predictions[i] == 0:
            cf_matrix[0][0] += 1  # true negative
        elif ground_truth[i] == predictions[i] == 1:
            cf_matrix[1][1] += 1  # true positive
        elif ground_truth[i] == 0 and predictions[i] == 1:
            cf_matrix[0][1] += 1  # false positive
        else:
            cf_matrix[1][0] += 1  # false negative

    if cf_matrix[1][1] + cf_matrix[0][1] != 0:
        precision = cf_matrix[1][1] / (cf_matrix[1][1] + cf_matrix[0][1]) * 100
    else:
        precision = 0

    if cf_matrix[1][1] + cf_matrix[1][0] != 0:
        recall = cf_matrix[1][1] / (cf_matrix[1][1] + cf_matrix[1][0]) * 100
    else:
        recall = 0

    if precision != 0 and recall != 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0

    ground_truth = np.array(ground_truth).astype(int)
    predictions = np.array(predictions).astype(int)

    fpr, tpr, threshold = metrics.roc_curve(ground_truth, predictions)
    roc_auc = metrics.auc(fpr, tpr)

    metrics_dict = {
        'stage': stage,
        'accuracies': aux_metrics['accuracies'],
        'losses': aux_metrics['losses'],
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': cf_matrix,
        'threshold': threshold,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }

    return metrics_dict


#	function for neural network training 
def train(net, device, train_loader, optimizer, criterion, aux_metrics_train):
    net.train()

    train_loss = 0.0
    total = 0
    correct = 0

    pneumonia_samples = 0
    healthy_samples = 0

    inner = tqdm.tqdm(total=len(train_loader.dataset), desc='Train samples', position=3)
    acc_log = tqdm.tqdm(total=0, position=4, bar_format='{desc}')
    loss_log = tqdm.tqdm(total=0, position=5, bar_format='{desc}')
    distribution_log = tqdm.tqdm(total=0, position=6, bar_format='{desc}')

    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.type(torch.LongTensor).to(device)

        optimizer.zero_grad()
        outputs = net(images)

        loss = criterion(outputs, targets)

        train_loss += loss.item()
        _, predicted = outputs.max(1)

        for target in targets:
            if target == 1:
                pneumonia_samples += 1
            else:
                healthy_samples += 1

        total_samples = pneumonia_samples + healthy_samples
        pneumonia_percentage = pneumonia_samples / total_samples * 100
        healthy_percentage = healthy_samples / total_samples * 100

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        for i in range(len(predicted)):
            aux_metrics_train['predictions'].append(predicted[i])
            aux_metrics_train['ground_truth'].append(targets[i])

        loss.backward()
        optimizer.step()

        acc = 100. * correct / total

        acc_log.set_description_str('Train accuracy: {:.2f}%'.format(acc))
        loss_log.set_description_str('Train loss: {:.4f}'.format(train_loss / (batch_idx + 1)))
        distribution_log.set_description_str(
            'Pneumonia / Healthy: {:.2f}% / {:.2f}%'.format(pneumonia_percentage, healthy_percentage))

        inner.update(images.shape[0])

    aux_metrics_train['losses'].append(train_loss / (batch_idx + 1))
    aux_metrics_train['accuracies'].append(acc)


#	function for neural network validation
def validate(net, device, val_loader, criterion, path, aux_metrics_val):
    net.eval()

    val_loss = 0.0
    total = 0
    correct = 0

    pneumonia_samples = 0
    healthy_samples = 0

    inner = tqdm.tqdm(total=len(val_loader.dataset), desc='Validation samples', position=8)
    acc_log = tqdm.tqdm(total=0, position=9, bar_format='{desc}')
    loss_log = tqdm.tqdm(total=0, position=10, bar_format='{desc}')
    distribution_log = tqdm.tqdm(total=0, position=11, bar_format='{desc}')

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.type(torch.LongTensor).to(device)
            outputs = net(images)

            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for target in targets:
                if target == 1:
                    pneumonia_samples += 1
                else:
                    healthy_samples += 1

            total_samples = pneumonia_samples + healthy_samples
            pneumonia_percentage = pneumonia_samples / total_samples * 100
            healthy_percentage = healthy_samples / total_samples * 100

            for i in range(len(predicted)):
                aux_metrics_val['predictions'].append(predicted[i])
                aux_metrics_val['ground_truth'].append(targets[i])

            acc = 100. * correct / total

            acc_log.set_description_str('Validation accuracy: {:.2f}%'.format(acc))
            loss_log.set_description_str('Validation loss: {:.4f}'.format(val_loss / (batch_idx + 1)))
            distribution_log.set_description_str(
                'Pneumonia / Healthy: {:.2f}% / {:.2f}%'.format(pneumonia_percentage, healthy_percentage))

            inner.update(images.shape[0])

    aux_metrics_val['losses'].append(val_loss / (batch_idx + 1))
    aux_metrics_val['accuracies'].append(acc)

    aux_metrics_val['ground_truth'] = np.array(aux_metrics_val['ground_truth']).astype(int)
    aux_metrics_val['predictions'] = np.array(aux_metrics_val['predictions']).astype(int)

    fpr, tpr, threshold = metrics.roc_curve(aux_metrics_val['ground_truth'], aux_metrics_val['predictions'])
    roc_auc = metrics.auc(fpr, tpr)

    if roc_auc > aux_metrics_val['best_val_auc']:
        aux_metrics_val['best_val_auc'] = roc_auc
        checkpoint = {
            'state_dict': net.state_dict(),
        }

        torch.save(checkpoint, path)


#	function for neural network testing
def test(net, device, test_loader, criterion, aux_metrics_test):
    net.eval()

    test_loss = 0.0
    total = 0
    correct = 0

    pneumonia_samples = 0
    healthy_samples = 0

    inner = tqdm.tqdm(total=len(test_loader.dataset), desc='Test samples', position=0)
    acc_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}')
    loss_log = tqdm.tqdm(total=0, position=2, bar_format='{desc}')
    distribution_log = tqdm.tqdm(total=0, position=3, bar_format='{desc}')

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.type(torch.LongTensor).to(device)
            outputs = net(images)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for target in targets:
                if target == 1:
                    pneumonia_samples += 1
                else:
                    healthy_samples += 1

            total_samples = pneumonia_samples + healthy_samples
            pneumonia_percentage = pneumonia_samples / total_samples * 100
            healthy_percentage = healthy_samples / total_samples * 100

            acc = 100. * correct / total

            acc_log.set_description_str('Test accuracy: {:.2f}%'.format(acc))
            loss_log.set_description_str('Test loss: {:.4f}'.format(test_loss / (batch_idx + 1)))
            distribution_log.set_description_str(
                'Pneumonia / Healthy: {:.2f}% / {:.2f}%'.format(pneumonia_percentage, healthy_percentage))

            inner.update(images.shape[0])

            for i in range(len(predicted)):
                aux_metrics_test['predictions'].append(predicted[i])
                aux_metrics_test['ground_truth'].append(targets[i])

    aux_metrics_test['losses'].append(test_loss / (batch_idx + 1))
    aux_metrics_test['accuracies'].append(acc)


#	function that calculates the weights according to the class distribution (in percents)
def calculate_weights(distribution, device):
    weights = torch.tensor(distribution, dtype=torch.float32)
    weights = weights.to(device)
    weights = weights / weights.sum()
    weights = 1.0 / weights
    weights = weights / weights.sum()

    return weights


#	function for testing certain model 
def test_net(source_dataset_name, test_dataset_name, csv_file, CustomDataset, path_to_net, path_to_save):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    criterion = nn.CrossEntropyLoss()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms = transform.Compose([
        transform.Resize((224, 224)),
        transform.ToTensor(),
        transform.Normalize(mean, std)
    ])

    if test_dataset_name == 'tb':
        dataset = CustomDataset(csv_file, transforms, True)
    else:
        dataset = CustomDataset(csv_file, transforms)

    net = torchvision.models.resnet50(pretrained=True)
    num_ftrs = net.fc.in_features

    for param in net.parameters():
        param.requires_grad = False

    net.fc = nn.Linear(num_ftrs, 2)

    state_dict = torch.load(path_to_net)
    net.load_state_dict(state_dict['state_dict'])
    net = net.to(device)

    dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

    aux_metrics_test = {
        'losses': [],
        'accuracies': [],
        'predictions': [],
        'ground_truth': []
    }

    metrics_test = []

    net.eval()

    test(net, device, dataloader, criterion, aux_metrics_test)

    metrics_test = calculate_metrics(aux_metrics_test, 'Test')
    show_metrics(metrics_test)

    report = {
        'test_metrics': metrics_test,
        'source_dataset': source_dataset_name,
        'test_dataset': test_dataset_name
    }

    torch.save(report, path_to_save)