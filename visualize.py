import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


#	function that shows 'number_images' images from 'data_loader'
def show_image(data_loader, number_images):
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    images = torch.Tensor(images)
    fig = plt.figure(figsize=(number_images, 4))
    for idx in np.arange(number_images):
        ax = fig.add_subplot(2, number_images / 2, idx + 1, xticks=[], yticks=[])
        img = np.transpose(images[idx])
        plt.imshow(img, cmap='gray')
        plt.title(str(labels[idx]))


#	function that shows the metrics
def show_metrics(metrics_dict):
    stage = metrics_dict['stage']
    accuracies = metrics_dict['accuracies']
    losses = metrics_dict['losses']
    precision = metrics_dict['precision']
    recall = metrics_dict['recall']
    f1_score = metrics_dict['f1_score']
    cf_matrix = metrics_dict['confusion_matrix']
    fpr = metrics_dict['fpr']
    tpr = metrics_dict['tpr']
    roc_auc = metrics_dict['roc_auc']

    cf_matrix = np.asarray(cf_matrix).reshape(2, 2)

    group_names = ['True Healthy', 'False Pneumonia', 'False Healthy', 'True Pneumonia']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure()
    plot = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plot.set(xlabel='Predicted label', ylabel='True label')
    plot.set_title(
        'Epoch: {}\n{} accuracy: {:.2f}%\n{} loss: {:.4f}\nPrecision: {:.2f}%\nRecall: {:.2f}%\nF1 score: {:.2f}%\nAUC score: {:.4f}\nConfusion matrix:'.format(
            len(losses), stage, accuracies[len(accuracies) - 1], stage, losses[len(losses) - 1], precision, recall,
            f1_score, roc_auc), fontsize=15)

    plt.show()


#	function that shows class distribution from .csv file
def show_distribution(csv):
    pneumonia_samples = len(csv[csv['Target'] == 1])
    healthy_samples = len(csv[csv['Target'] == 0])

    proportions = [healthy_samples, pneumonia_samples]
    labels = ['Healthy', 'Pneumonia']
    colors = ['G', 'R']
    explode = (0.2, 0.2)

    plt.pie(proportions, labels=labels, colors=colors,
            startangle=20, shadow=True,
            radius=2.5, autopct='%1.1f%%', explode=explode)

    plt.legend()
    plt.show()