#Import necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.nn import GCNConv


class TemporalPositionalEncoding(nn.Module):
    """
    Definition:
    A temporal Positional Encoding module is applied to the node features before the convolutional layers.
    This helps the model learn the temporal order of the input data. To do so, a small bias is added to the
    input features, so the model learns to disinguish between the time steps. 

    The number of input channels is used to create a torch parameter of the same size that stores the biases. 

    Arguments:
    arg[1] : in_channels(int) - size of the input channels 
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, in_channels))
        nn.init.uniform_(self.bias, -0.02, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""


class ChebGCN(nn.Module):
    def __init__(self, input_channel, hidden_layers, output_channel, K, normalization = 'sym'):
        """
        Definition:
        Creating the spectral graph convolutional network using the Cheby-chev pytorch package for forecasting forward in time.

        Arguments:
        arg[1] : Input channels (int) - size of each input channel
        arg[2] : Hidden layers (int) - size of number of hidden channels
        arg[3] : Output channels (int) - size of each output channel
        arg[4] : K (int) - size of the ChebyChev polynomial degree
        arg[5] : Normalization (str) - default is symmetrical. Other to choose from is 'None'
        """

        super().__init__()
        self.tpe   = TemporalPositionalEncoding(input_channel)
        self.conv1 = ChebConv(input_channel, hidden_layers, K, normalization=normalization)
        self.conv2 = ChebConv(hidden_layers, hidden_layers, K, normalization=normalization)
        self.conv3 = ChebConv(hidden_layers, output_channel, K, normalization=normalization)

    def forward(self, x, edge_index, lambda_max, edge_weight = None):
        x = self.tpe(x)
        x = self.conv1(x, edge_index, edge_weight = edge_weight, lambda_max = lambda_max)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_weight = edge_weight, lambda_max = lambda_max)
        x = F.relu(x)

        x = self.conv3(x, edge_index, edge_weight = edge_weight, lambda_max = lambda_max)
        return x

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

class SimpleGCN(nn.Module):
    def __init__(self, input_channel, hidden_layers, output_channel):
        """
        Definition:
        Creating a Graph Convolutional Network using the built in GCN module from Pytorch named GCNCVonv.
        This a more exact representation of the method used in (Kipf & Welling, 2017) with a few
        simplifications of the models compared to using the chebychev.


        Arguments:
        arg[1] : Input channels (int) - size of each input channel
        arg[2] : Hidden layers (int) - size of number of hidden channels
        arg[3] : Output channels (int) - size of each output channel
        """
        super().__init__()
        self.tpe   = TemporalPositionalEncoding(input_channel)
        self.conv1 = GCNConv(input_channel, hidden_layers)
        self.conv2 = GCNConv(hidden_layers, hidden_layers)
        self.conv3 = GCNConv(hidden_layers, output_channel)

    def forward(self, x, edge_index):
        x = self.tpe(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        return x

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
#residuals + deeper model (previously used was three convolutions)

class ResidualsChebGCN(nn.Module):
    def __init__(self, input_channel, hidden_layers, output_channel, K, normalization = 'sym'):
        super().__init__()

        """
        Definition:
        Creating a residual block to be used for the ChebGCN model with deeper layers.
        The residual blocks are used for creating deeper ML models: 
        sources used:
        https://docs.pytorch.org/docs/2.12/generated/torch.nn.BatchNorm1d.html
        https://www.geeksforgeeks.org/deep-learning/what-is-batch-normalization-in-deep-learning/

        Batch normalization is used here to reduce:
            - Vanishing gradients 
            - Provides regularizations in the deeper model
            - Stabilizes the model and accelerates convergence during the training process 
            (inspired by ResNet models)

        Arguments:
        arg[1] : Input channels (int) - size of each input channel
        arg[2] : Hidden layers (int) - size of number of hidden channels
        arg[3] : Output channels (int) - size of each output channel
        arg[4] : K (int) - size of the ChebyChev polynomial degree
        arg[5] : Normalization (str) - default is symmetrical. Other to choose from is 'None'
        """

        self.conv1 = ChebConv(input_channel, hidden_layers, K, normalization=normalization)
        self.conv2 = ChebConv(hidden_layers, hidden_layers, K, normalization=normalization)
        #Batch normalizations - normalising the data within each batch 
        self.bn1 = nn.BatchNorm1d(hidden_layers) 
        self.bn2 = nn.BatchNorm1d(hidden_layers)
        self.skip = nn.Linear(input_channel, hidden_layers, bias = False) if input_channel != output_channel else nn.Identity()

    def forward(self, x, edge_index, lambda_max, edge_weight = None):

        """
        Definition:
        Applying the ReLu activation functions. 
        """

        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x, edge_index, lambda_max=lambda_max, edge_weight=edge_weight)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, lambda_max=lambda_max, edge_weight=edge_weight)))
        residual_x = F.relu(x + residual)
        return residual_x

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

class DeepChebGCN(nn.Module):
    def __init__(self, input_channel, hidden_layers, output_channel, K, normalization = 'sym', block_numbers = 3):
        super().__init__()

        """
        Definition: 
        Model architecture for a deeper ChebGCN model. 
        Uses the residuals and batch normalizations from Residuals ChebGCN 

        Arguments:
        arg[1] : Input channels (int) - size of each input channel
        arg[2] : Hidden layers (int) - size of number of hidden channels
        arg[3] : Output channels (int) - size of each output channel
        arg[4] : K (int) - size of the ChebyChev polynomial degree
        arg[5] : Normalization (str) - default is symmetrical. Other to choose from is 'None'
        arg[6] : Block numbers (int) - default is 3 -> 6 layers (two convolutional layers applied per block)
        """

        self.tpe = TemporalPositionalEncoding(input_channel)
        self.input_proj = nn.Linear(input_channel, hidden_layers) #Linear transformation
        self.blocks = nn.ModuleList([ResidualsChebGCN(hidden_layers, hidden_layers, hidden_layers, K, normalization) for _ in range(block_numbers)])
        self.output_proj = nn.Linear(hidden_layers, output_channel)

    def forward(self, x, edge_index, lambda_max, edge_weight = None):
        x = self.tpe(x)
        x = F.relu(self.input_proj(x))
        for block in self.blocks:
            x = block(x, edge_index, lambda_max, edge_weight=edge_weight)
        return self.output_proj(x)

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#edge features - uses the distance between the edges
class EdgeWeightChebGCN(nn.Module):

    def __init__(self, input_channel, hidden_layers, output_channel, K, normalization = 'sym'):
        super().__init__()

        """
        Definition: 
        This ChebGCN model use weights to determine the connection weight between nodes. 
        The weights follow a Gaussian function calculated from the distance between the connected nodes in the graph structure
        
        Arguments:
        arg[1] : Input channels (int) - size of each input channel
        arg[2] : Hidden layers (int) - size of number of hidden channels
        arg[3] : Output channels (int) - size of each output channel
        arg[4] : K (int) - size of the ChebyChev polynomial degree
        arg[5] : Normalization (str) - default is symmetrical. Other to choose from is 'None'
        """

        self.tpe   = TemporalPositionalEncoding(input_channel)
        self.conv1 = ChebConv(input_channel, hidden_layers, K, normalization=normalization)
        self.conv2 = ChebConv(hidden_layers, hidden_layers, K, normalization=normalization)
        self.conv3 = ChebConv(hidden_layers, output_channel, K, normalization=normalization)

    def forward(self, x, edge_index, lambda_max, edge_weight = None):
        x = self.tpe(x)
        x = self.conv1(x, edge_index, lambda_max=lambda_max, edge_weight = edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, lambda_max = lambda_max, edge_weight = edge_weight)
        x = F.relu(x)
        x = self.conv3(x, edge_index, lambda_max = lambda_max, edge_weight = edge_weight)
        return x
