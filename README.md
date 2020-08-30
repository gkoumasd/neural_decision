# Deep Multimodal Fusion of Uni-modal Classifers

This is an implementation of the paper "Deep Multimodal Fusion for Persuasiveness Prediction" [1] in PyTorch.
However, the current implementation is a unified framework supporting many tasks, datasets, and neural models.



## Instructions to run the code

### Download the datasets
We trained uni-modal classifiers (Linguistic, Visual, Acoustic) for two benchmark corpora video sentiment analysis datasets, CMU-MOSEI and CMU-MOSI. We exported the uni-modal sentiment judgments to pickle files, stored at data/pickle. 


### Do A Single Run (train/valid/test) 

1. Set up the configurations in config/run.ini
2. python run.py -config config/run.ini

#### Configuration setup
+ Monologue
  + **mode = run**
  + **pickle_dir_path = /path/to/datasets/**. The absolute path of the folder storing the datasets.
  + **dataset_name in `{'cmumosei','cmumosi'}`**. Name of the dataset.
  + **label in `{'sentiment'}`**. Multiple labels should be joined by ','. 
  + **embedding_trainable in `{'True','False'}`**. Whether you want to train the word embedding for textual modality. Usually set to be True.
  + **model specific parameters**. For running a model on the dataset, uncomment the respective area of the model and comment the areas for the other models. Please refer to the model implementations in /models/ for the meaning of each model specific parameter.
    + supported models include but are not limited to:
      + SimpleNN
     
### Grid Search for the Best Parameters
1. Set up the configurations in config/grid_search.ini. Tweak a couple of fields in the single run configurations, as instructed below.
2. Write up the hyperparameter pool in config/grid_parameters/.
3. python run.py -config config/grid_search.ini

#### Configuration setup
+ **mode = run_grid_search**
+ **grid_parameters_file**. The name of file storing the parameters to be searched, under the folder /config/grid_parameters. 
  + the format of a file is:
    + [COMMON]
    + var_1 = val_1;val_2;val_3
    + var_2 = val_1;val_2;val_3
+ **search_times**. The number of times the program searches in the pool of parameters.
+ **output_file**.  The file storing the performances for each search in the pool of parameters. By default, it is eval/grid_search_`{dataset_name}`_`{network_type}`.csv




Refferences:


[1]. Nojavanasghari, B., Gopinath, D., Koushik, J., Baltrušaitis, T., & Morency, L. P. (2016, October). Deep multimodal fusion for persuasiveness prediction. In Proceedings of the 18th ACM International Conference on Multimodal Interaction (pp. 284-288).
