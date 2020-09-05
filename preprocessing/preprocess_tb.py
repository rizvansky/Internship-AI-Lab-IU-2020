train_val_csv = pd.DataFrame(columns=['Path to img', 'Target'])
test_csv = pd.DataFrame(columns=['Path to img', 'Target'])
df = pd.DataFrame(columns=['Path to img', 'Target'])

train_norm_glob = glob('/datasets/medical/tb/train/norm/*/*.bin')
train_pneu_glob = glob('/datasets/medical/tb/train/pathology/*/*.bin')

test_norm_glob = glob('/datasets/medical/tb/test/norm/*/*.bin')
test_pneu_glob = glob('/datasets/medical/tb/test/pathology/*/*.bin')

for path in train_norm_glob:
    new_row = {'Path to img' : path, 'Target' : 0}
    train_val_csv = train_val_csv.append(new_row, ignore_index=True)
    df = df.append(new_row, ignore_index=True)
    
for path in train_pneu_glob:
    new_row = {'Path to img' : path, 'Target' : 1}
    train_val_csv = train_val_csv.append(new_row, ignore_index=True)
    df = df.append(new_row, ignore_index=True)
    
for path in test_norm_glob:
    new_row = {'Path to img' : path, 'Target' : 0}
    test_csv = test_csv.append(new_row, ignore_index=True)
    df = df.append(new_row, ignore_index=True)
    
for path in test_pneu_glob:
    new_row = {'Path to img' : path, 'Target' : 1}
    test_csv = test_csv.append(new_row, ignore_index=True)
    df = df.append(new_row, ignore_index=True)
	
train_csv, val_csv = train_test_split(train_val_csv, test_size=0.111, shuffle=True)

train_csv.to_csv('tb/test.csv')
val_csv.to_csv('tb/val.csv')
test_csv.to_csv('tb/test.csv')
df.to_csv('tb/df.csv')