data_path = '/datasets/medical/rsna'
df = pd.read_csv(os.path.join(data_path, 'stage_2_train_labels.csv'))
df.drop_duplicates(subset='patientId', keep='first', inplace=True) 
df.index = list(range(len(df)))

csv = pd.DataFrame(columns=['Path to img', 'Target'])

for i in range(len(df)):
    if df['Target'][i] == 1:
        path = df['patientId'][i] + '.dcm'
        new_row = {'Path to img' : os.path.join(data_path, 'stage_2_train_images', path), 'Target' : 1}
        csv = csv.append(new_row, ignore_index=True)
        
for i in range(len(df)):
    if df['Target'][i] == 0:
        path = df['patientId'][i] + '.dcm'
        new_row = {'Path to img' : os.path.join(data_path, 'stage_2_train_images', path), 'Target' : 0}
        csv = csv.append(new_row, ignore_index=True)


df_balanced = pd.DataFrame(columns=['Path to img', 'Target'])
pneumonia_count = 0

df = csv
for i in range(len(df)):
    if df['Target'][i] == 1:
        path = df['Path to img'][i]
        new_row = {'Path to img' : path, 'Target' : 1}
        df_balanced = df_balanced.append(new_row, ignore_index=True)
        pneumonia_count += 1
        
healthy_count = 0
for i in range(len(df)):
    if df['Target'][i] == 0:
        path = df['Path to img'][i]
        new_row = {'Path to img' : path, 'Target' : 0}
        df_balanced = df_balanced.append(new_row, ignore_index=True)
        healthy_count += 1

    if healthy_count == pneumonia_count:
        break

df.to_csv('rsna/df.csv')
df_balanced.to_csv('rsna/df_balanced.csv')

train_csv, val_test_csv = train_test_split(df_balanced, test_size=0.2)
val_csv, test_csv = train_test_split(val_test_csv, test_size=0.5)

train_csv.index = range(len(train_csv))  
val_csv.index = range(len(val_csv)) 
test_csv.index = range(len(test_csv)) 

train_csv.to_csv('rsna/train.csv')
val_csv.to_csv('rsna/val.csv')
test_csv.to_csv('rsna/test.csv')