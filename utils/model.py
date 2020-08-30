import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import time
import data as chooseReader
import pandas as pd
import numpy as np
import os
import copy
from utils.eveluation import *

def train(params, model):
    
    
    criterion = get_criterion(params)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    
    
    # Temp file for storing the best model 
    temp_file_name = str(int(np.random.rand()*int(time.time())))
    params.best_model_file = os.path.join('tmp',temp_file_name)
    
    train_generator, test_generator = chooseReader.setup(params)
    best_test_loss = 99999
    for i in range(params.epochs):
         print('epoch: ', i)
         model.train()
         
         with tqdm(total = params.batch_size*len(train_generator)) as pbar:
             time.sleep(0.05)    
         
             # Training
             for _i, _data in enumerate(train_generator):
                #_i,_data = next(iter(enumerate(train_generator)))
                
                x_train = _data[:,:-1].to(params.device)
                y_train = _data[:,-1].to(params.device)
                
                optimizer.zero_grad()
                predictions = model(x_train).squeeze(-1)
                
                loss = get_loss(params,criterion, predictions, y_train)
                
                loss.backward()
                
                nn.utils.clip_grad_norm_(model.parameters(), params.clip)
                
                optimizer.step()
                
                # Compute Training Accuracy 
                n_total = len(predictions)
                if params.label == 'sentiment':
                    n_correct = ((predictions>=0.5).type(torch.float) == y_train).sum().item()
                train_acc = n_correct/n_total     
                
                #Update Progress Bar
                pbar.update(params.batch_size)
                ordered_dict={'acc': train_acc, 'loss':loss.item()}        
                pbar.set_postfix(ordered_dict=ordered_dict)
                
                
         #################### Compute Validation Performance##################       
         model.eval()   
         
         with torch.set_grad_enabled(False):
              pred_test, y_test = get_predictions(model,params, test_generator)
              test_loss = get_loss(params,criterion, pred_test, y_test)
              print('Test Performance:')
              performances = evaluate(params,pred_test,y_test)
              print('val_acc = {}, val_loss = {}'.format(performances['acc'], test_loss))
              
              if test_loss < best_test_loss:
                  best_test_loss = test_loss
                  torch.save(model,params.best_model_file)
                  print('The best model up till now. Saved to File.')

def get_predictions(model, params, data_generator):    
    predictions = []
    targets = []
    
    for __i, __data in enumerate(data_generator):
       X = __data[:,:-1].to(params.device)
       y = __data[:,-1].to(params.device)
       y_hat = model(X).squeeze(-1)
       
       predictions.append(y_hat.detach())
       targets.append(y.detach())
       
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    
    return predictions, targets
    
def get_criterion(params):
    if params.dataset_name=='cmu_mosi' or params.dataset_name=='cmu_mosei':
        #Binary Cross Entropy
         criterion = nn.BCELoss()
    return criterion 

def get_loss(params, criterion, predictions, y):
    
    if (params.label == 'sentiment'):
        #Predictions should be passed through Sigmoid function
        _loss = criterion(predictions,y)
      
        return _loss
    
def test(model,params):
    model.eval()
    test_output,test_target= get_predictions(model, params, split = 'test')    
    
         
         
    performances = evaluate(params,test_output,test_target)
    return performances    


def print_performance(performance_dict, params):
    performance_str = ''
    if params.label == 'sentiment':
        for key, value in performance_dict.items():
            performance_str = performance_str+ '{} = {} '.format(key,value)
    print(performance_str)
    return performance_str


    
def save_performance(params, performance_dict):
        
    df = pd.DataFrame()
    output_dic = {'dataset' : params.dataset_name,
                    'network' : params.network_type,
                    'model_dir_name': params.dir_name}
    output_dic.update(performance_dict)
    df = df.append(output_dic, ignore_index = True)

    if not 'output_file' in params.__dict__:
        params.output_file = 'eval/{}_{}.csv'.format(params.dataset_name, params.network_type)
    print('CSV:',params.output_file)
    df.to_csv(params.output_file, encoding='utf-8', index=True)   
    
    
#Save the model
def save_model(model,params,performance_str):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
        
    #Create model dir
    params.dir_name = str(round(time.time()))
    dir_path = os.path.join('tmp',params.dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
    #save the learned model
    torch.save(model.state_dict(),os.path.join(dir_path,'model'))
    params.export_to_config(os.path.join(dir_path,'config.ini'))
    

    
    #Write performance string
    eval_path = os.path.join(dir_path,'eval')
    with open(eval_path,'w') as f:
        f.write(performance_str)    
            

