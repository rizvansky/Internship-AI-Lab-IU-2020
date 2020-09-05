chest_14_folder = '/datasets/medical/chest-14'

df = pd.read_csv(os.path.join(chest_14_folder, 'Data_Entry_2017.csv'))

my_glob = glob(os.path.join(chest_14_folder, 'images*/images/*.png'))

full_img_paths = {os.path.basename(x): x for x in my_glob}
df['full_path'] = df['Image Index'].map(full_img_paths.get)

df['Pneumonia'] = df['Finding Labels'].map(lambda result: 1 if 'Pneumonia' in result else 0)
    
df['Target'] = df.apply(lambda target: [target['Pneumonia']], 1).map(lambda target: target[0])

df.index = range(len(df))

csv = pd.DataFrame(columns=['Path to img', 'Target'])

for i in range(len(df)):
    if df['Pneumonia'][i] == 1:
        new_row = {'Path to img' : df['full_path'][i], 'Target' : 1}
        csv = csv.append(new_row, ignore_index=True)
        
for i in range(len(df)):
    if df['Finding Labels'][i] == 'No Finding':
        new_row = {'Path to img' : df['full_path'][i], 'Target' : 0}
        csv = csv.append(new_row, ignore_index=True)
		
df = csv
df.index = range(len(df))
df_balanced = pd.DataFrame(columns=['Path to img', 'Target'])

pneumonia_count = 0		
for i in range(len(df)):
    if df['Target'][i] == 1:
        new_row = {'Path to img' : df['Path to img'][i], 'Target' : 1}
        df_balanced = df_balanced.append(new_row, ignore_index=True)
        pneumonia_count += 1
        
healthy_count = 0
for i in range(len(df)):
    if df['Target'][i] == 0:
        new_row = {'Path to img' : df['Path to img'][i], 'Target' : 0}
        df_balanced = df_balanced.append(new_row, ignore_index=True)
        healthy_count += 1
        
    if healthy_count == pneumonia_count:
        break
        
df_balanced.index = range(len(df_balanced))        
df.to_csv('chest14/df.csv')
df_balanced.to_csv('chest14/df_balanced.csv')

train_csv, val_test_csv = train_test_split(df_balanced, test_size=0.2)
val_csv, test_csv = train_test_split(val_test_csv, test_size=0.5)

train_csv.index = range(len(train_csv))  
val_csv.index = range(len(val_csv)) 
test_csv.index = range(len(test_csv)) 

train_csv.to_csv('chest14/train.csv')
val_csv.to_csv('chest14/val.csv')
test_csv.to_csv('chest14/test.csv')