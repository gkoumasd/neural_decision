import torch
import argparse
from utils.params import Params
import data as chooseReader
import random
import models
from utils.model import *
from utils.eveluation import *
from utils.io_utils import parse_grid_parameters


def run(params, data_generator):
    model = models.setup(params).to(params.device)
    
    print('Training the model!')
    train(params,model)
    model = torch.load(params.best_model_file)
    os.remove(params.best_model_file)
    
    pred, y = get_predictions(model,params, data_generator)
    performance_dict  = evaluate(params,pred,y)
        
    performance_str = print_performance(performance_dict, params)
    save_model(model,params,performance_str)
    
     
    return performance_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='running experiments')
    parser.add_argument('-config', action = 'store', dest = 'config_file', help = 'please enter configuration file.',default = 'config/run.ini')
    args = parser.parse_args()
    params = Params()
    params.parse_config(args.config_file) 
    params.config_file = args.config_file
    mode = 'run'
    if 'mode' in params.__dict__:
        mode = params.mode 
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    data_generators = chooseReader.setup(params)
    
    
    
    if mode == 'run':
        
        performance_dict = run(params,data_generators[1])
        save_performance(params, performance_dict)
       
    elif mode == 'run_grid_search':    
        print('Grid Search Begins.')
        if not 'grid_parameters_file' in params.__dict__:
            params.grid_parameters_file = params.network_type+'.ini'
            
        grid_parameters = parse_grid_parameters(os.path.join('config','grid_parameters',params.grid_parameters_file))
            
        df = pd.DataFrame()
            
        if not 'output_file' in params.__dict__:
            params.output_file = 'eval/grid_search_{}_{}.csv'.format(params.dataset_name, params.network_type)
                
        for i in range(params.search_times):
            parameter_list = []
            merged_dict = {}
            for key in grid_parameters:
                value = random.choice(grid_parameters[key])
                parameter_list.append((key, value))
                merged_dict[key] = value
            params.setup(parameter_list)
                
            performance_dict = run(params,data_generators[1])
            performance_dict['model_dir_name'] = params.dir_name
            merged_dict.update(performance_dict)
            df = df.append(merged_dict, ignore_index=True)      
            df.to_csv(params.output_file, encoding='utf-8', index=True)
    else:
        print('wrong input run mode!')
        exit(1)
                
                
      