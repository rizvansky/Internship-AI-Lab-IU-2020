datasets = ['chest_xray_pneumonia', 'rsna', 'tb', 'gb7', 'chest14', 'all']

for source_dataset in datasets:
    for test_dataset in datasets:
        df = None
		if source_dataset != test_dataset and source_dataset != 'all':
			df = pd.read_csv(os.path.join('/home/intern/r.iskaliev/work', test_dataset, 'df.csv'))
		else:
			df = pd.read_csv(os.path.join('/home/intern/r.iskaliev/work', test_dataset, 'test.csv'))
    
		path_to_net = os.path.join('/home/intern/r.iskaliev/work', source_dataset, 'net.pth')
		path_to_save = os.path.join('/home/intern/r.iskaliev/work', source_dataset, 'test_on_' + test_dataset + '.pth')
		test_net(source_dataset, test_dataset, df, XRayDatasetFromCSV, path_to_net, path_to_save)

auc_matrix = np.random.rand(len(datasets), len(datasets))

for i in range(len(datasets)):
    for j in range(len(datasets)):
        metrics = torch.load(os.path.join('/home/intern/r.iskaliev/work', datasets[i], 'test_on_' + datasets[j] + '.pth'))['test_metrics']
        auc_matrix[j][i] = metrics['roc_auc']

plt.figure(figsize=(16, 13))
sns.set(font_scale=2.5)
heatmap = sns.heatmap(auc_matrix, cbar_kws={'label': 'AUC score'}, cmap='Purples', xticklabels=datasets, yticklabels=datasets,  linewidths=1, linecolor='black')
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)
heatmap.axhline(y=0, color='k',linewidth=5)
heatmap.axhline(y=6, color='k',linewidth=5)
heatmap.axvline(x=0, color='k',linewidth=5)
heatmap.axvline(x=6, color='k',linewidth=5)

plt.show()

chart = {
    'heatmap_array': heatmap_array,
    'heatmap': heatmap
}

torch.save(chart, 'chart.pth')		
		