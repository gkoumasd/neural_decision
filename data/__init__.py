def setup(opt):
   
     if opt.dataset_name.lower() == 'cmu_mosi' or opt.dataset_name.lower() == 'cmu_mosei':
         from data.reader import DataReader
         
     reader = DataReader(opt)
     reader = reader.prepare_data()
     
     return reader    
 
