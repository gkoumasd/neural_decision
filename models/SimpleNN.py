from torch import nn
import torch.nn.functional as F
import torch

class SimpleNN(nn.Module):
    
    def __init__(self,opt):
        super(SimpleNN, self).__init__()
        
        self.input = opt.input_size
        self.drop_out = opt.drop_out
        self.embed_dim = opt.embed_dim
        self.hidden_dim = opt.hidden_dim
        self.output = opt.output
        
       
        
        self.fc_in = nn.Linear(self.input, self.embed_dim) 
        self.fc_hiddens = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim) \
                                   for _dim in range(self.hidden_dim)])
        self.fc_out = nn.Linear(self.embed_dim ,self.output)
        
    def forward(self, x):
        x = F.relu(self.fc_in(F.dropout(x,self.drop_out)))
        for fc_hidden in self.fc_hiddens:
            x = F.relu(fc_hidden(x))
        x = torch.sigmoid(self.fc_out(x)) 
        return x   
