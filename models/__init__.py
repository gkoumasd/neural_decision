from models.SimpleNN import SimpleNN


def setup(opt):

    print("network type: " + opt.network_type)

    if opt.network_type == "simpleNN":
        model = SimpleNN(opt)
        
    return model    
