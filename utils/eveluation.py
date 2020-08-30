from sklearn.metrics import accuracy_score, f1_score
import torch


def evaluate(params, outputs, targets):
    
    outputs = (outputs>=0.5).type(torch.float)
    outputs_np = outputs.cpu().numpy()
    targets_np = targets.cpu().numpy()

    
    # Performances
    accuracy = accuracy_score(targets_np, outputs_np, normalize=True)
    
    f1 = f1_score(targets_np,outputs_np,average='weighted')
    
    performance_dict = {'acc':accuracy,'binary_f1':f1 }
    
    return performance_dict
    