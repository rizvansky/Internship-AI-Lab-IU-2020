normal_glob = glob('/datasets/medical/chest-xray-pneumonia/*/NORMAL/*.jpeg')
pneumonia_glob = glob('/datasets/medical/chest-xray-pneumonia/*/PNEUMONIA/*.jpeg')

pneumonia_glob_balanced = pneumonia_glob[0:len(normal_glob)]

df = pd.DataFrame(columns=['Path to img', 'Target'])
df_balanced = pd.DataFrame(columns=['Path to img', 'Target'])

for img_name in normal_glob:
    new_row = {'Path to img': img_name, 'Target': 0}
    df = df.append(new_row, ignore_index=True)
    df_balanced = df_balanced.append(new_row, ignore_index=True)

for img_name in pneumonia_glob:
    new_row = {'Path to img': img_name, 'Target': 1}
    df = df.append(new_row, ignore_index=True)
    
for img_name in pneumonia_glob_balanced:
    new_row = {'Path to img': img_name, 'Target': 1}
    df_balanced = df_balanced.append(new_row, ignore_index=True)
    
df.to_csv('chest_xray_pneumonia/df.csv')
df_balanced.to_csv('chest_xray_pneumonia/balanced.csv')

train_csv, val_test_csv = train_test_split(df_balanced, test_size=0.2)
val_csv, test_csv = train_test_split(val_test_csv, test_size=0.5)

train_csv.index = range(len(train_csv))
val_csv.index = range(len(val_csv))
test_csv.index = range(len(test_csv))

train_csv.to_csv('chest_xray_pneumonia/train.csv')
val_csv.to_csv('chest_xray_pneumonia/val.csv')
test_csv.to_csv('chest_xray_pneumonia/test.csv')