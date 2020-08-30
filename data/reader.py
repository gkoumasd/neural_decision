import pickle
import os
from torch.utils.data import DataLoader
import torch

class DataReader(object):
    
     def __init__(self,opt):
         self.data_path = opt.pickle_dir_path
         self.dataset_name = opt.dataset_name
         self.batch_size = opt.batch_size
         

         
     def prepare_data(self):
         data = self.load_pickle_data()
         
         #Train
         train = torch.from_numpy(data.loc[data['Dataset'] == 'Train', ['pos_l', 'neg_l','pos_v', 'neg_v','pos_a', 'neg_a', 'Label']].to_numpy()).float()
         train_generator = DataLoader(train, batch_size = self.batch_size, shuffle = True)
         
    
             
         
         #Test
         test = torch.from_numpy(data.loc[data['Dataset'] == 'Test', ['pos_l', 'neg_l','pos_v', 'neg_v','pos_a', 'neg_a', 'Label']].to_numpy()).float()
         test_generator = DataLoader(test, batch_size = self.batch_size, shuffle = False)
         
         
         
         return train_generator, test_generator
         
    
       

     def load_pickle_data(self):
        data = pickle.load(open(os.path.join(self.data_path,self.dataset_name) + '.pickle', 'rb'))
        return data
     