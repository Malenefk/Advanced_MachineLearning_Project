#Import packages
import json 

#Configuration file for architecture and training parameters
with open('config.json', 'r') as f:
    config = json.load(f)

    #Variable options: <psi>, <q>, <q_forcing_advective>
    variable = config['variable'] 

    window_length = config.get('window_length', 4) #Default winow is 4 - can be changed

    batch_size = config.get('batch_size', 32) #Default batch size is 32 - can be changed 

    epochs = config['epochs']

    hidden_layers = config.get('hidden_layers', 64) #Default value is 64 - can be changed

    learning_rate = config['lr']

    optimizer = config.get('optimizer', 'Adam')

    use_physical_constraint = config.get('use_physical_constraint', True)

    lambda_p = config['lambda_p'] #to use in the physical constraint loss function

    #Options are: 'ChebGCN', 'GCN', 'DeepChebGCN' and 'WeightedChebGCN'
    model_name = config['model_name'] 

    normalization = config.get('normalization', 'sym')

    K = config.get('K', 2) #Default is two - but is tested for other values for the ChebGCN models 

    path = config.get('path', '/Users/malenekarlsen/documents/Master/Vår26/MachineLearning/results') #need to figure out how to automatically name the files accordingly to the chosen metrics 

