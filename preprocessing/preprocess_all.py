work = '/home/intern/r.iskaliev/work'
chest_xray_pneumonia_balanced = pd.read_csv(os.path.join(work, 'chest_xray_pneumonia/df_balanced.csv'))
rsna_balanced = pd.read_csv(os.path.join(work, 'rsna/df_balanced.csv'))
tb_balanced = pd.read_csv(os.path.join(work, 'tb/df.csv'))
gb7_balanced = pd.read_csv(os.path.join(work, 'gb7/df.csv'))
chest14_balanced = pd.read_csv(os.path.join(work, 'chest14/df_balanced.csv'))
df_balanced = pd.concat([chest_xray_pneumonia_balanced, rsna_balanced, tb_balanced, gb7_balanced, chest14_balanced])

chest_xray_pneumonia = pd.read_csv(os.path.join(work, 'chest_xray_pneumonia/df.csv'))
rsna = pd.read_csv(os.path.join(work, 'rsna/df.csv'))
tb = pd.read_csv(os.path.join(work, 'tb/df.csv'))
gb7 = pd.read_csv(os.path.join(work, 'gb7/df.csv'))
chest14 = pd.read_csv(os.path.join(work, 'chest14/df.csv'))
df = pd.concat([chest_xray_pneumonia, rsna, tb, gb7, chest14])

df_balanced.to_csv('all/df_balanced.csv')
df.to_csv('all/df.csv')

train_csv, val_test_csv = train_test_split(df_balanced, test_size=0.2)
val_csv, test_csv = train_test_split(val_test_csv, test_size=0.5)

train_csv.index = range(len(train_csv))  
val_csv.index = range(len(val_csv)) 
test_csv.index = range(len(test_csv)) 

train_csv.to_csv('all/train.csv')
val_csv.to_csv('all/val.csv')
test_csv.to_csv('all/test.csv')