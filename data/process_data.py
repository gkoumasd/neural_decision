import numpy as np
import pandas as pd
import torch

import pickle


df = pd.read_csv('data/decesion_level_data_cmu_MOSEI.csv')
df.loc[df['Dataset'] == 'Valid', 'Dataset'] = 'Train'
df['ID'] = df.index
df.shape


#train = df[df['Dataset']=='Train']
#valid = df[df['Dataset']=='Valid']
#Concatenate train and valid dataset
#frames = [train, valid]
#train = pd.concat(frames)
#train = train.reset_index(drop=True)
#train.reset_index(drop=True, inplace=True)

#test = df[df['Dataset']=='Test']
#test = test.reset_index(drop=True)
#test.reset_index(drop=True, inplace=True)

#X_train = train.iloc[:,1:7]
#y_train = train.iloc[:,-1]

#X_test = test.iloc[:,1:7]
#y_test = test.iloc[:,-1]

#X_train = torch.from_numpy(X_train.to_numpy()).float()
#y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
#X_test = torch.from_numpy(X_test.to_numpy()).float()
#y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())


#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)

#with open('data/train_CMU_MOSI.pickle', 'wb') as f:
#    pickle.dump([X_train, y_train], f)


with open('data/pickle/cmu_mosei.pickle', 'wb') as f:
    pickle.dump(df, f)

