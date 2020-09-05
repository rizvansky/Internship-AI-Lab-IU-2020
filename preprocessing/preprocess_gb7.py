pneu_poli_paths = glob('/datasets/medical/GB7/Poliklinika/Pnevmoniya/*/*/IM*')
pneu_flg_paths = glob('/datasets/medical/GB7/FLG/MD__20190111-1231_pnevmoniya/*/IM*')
normal_flg_paths = glob('/datasets/medical/GB7/FLG/MD__20200424__norma/*/IM*')

pneumonia_paths = pneu_poli_paths + pneu_flg_paths
normal_paths = normal_flg_paths

print('Number of pneumonia samples: {}, number of normal samples: {}'.format(len(pneumonia_paths), len(normal_paths)))

df = pd.DataFrame(columns=['Path to img', 'Target'])

for path in pneumonia_paths:
    new_row = {'Path to img' : path, 'Target' : 0}
    df = df.append(new_row, ignore_index=True)
    
for path in normal_paths:
    new_row = {'Path to img' : path, 'Target' : 1}
    df = df.append(new_row, ignore_index=True)

train_csv, val_test_csv = train_test_split(csv, test_size=0.2, shuffle=True)
val_csv, test_csv = train_test_split(val_test_csv, test_size=0.5, shuffle=True)

train_csv.index = range(len(train_csv))
val_csv.index = range(len(val_csv))
test_csv.index = range(len(test_csv))

df.to_csv('gb7/df.csv')
train_csv.to_csv('gb7/train.csv')
val_csv.to_csv('gb7/val.csv')
test_csv.to_csv('gb7/train.csv')