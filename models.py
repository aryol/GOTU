import torch
from torch import nn
import numpy as np

# Random features model with ReLU activation
class RandomFeaturesReLU(nn.Module):
    def __init__(self, input_dimension):
        super(RandomFeaturesReLU, self).__init__()
        self.num_feature = 2 ** 13
        self.features = nn.Linear(input_dimension, self.num_feature)
        torch.nn.init.normal_(self.features.bias, mean=0, std=1/np.sqrt(input_dimension))
        torch.nn.init.normal_(self.features.weight, mean=0, std=1/np.sqrt(input_dimension))
        for param in self.features.parameters():
            param.required_grad = False
        
        self.weights = nn.Linear(self.num_feature, 1, bias=False)
        self.weights.weight.data.fill_(0.0)
    
    def forward(self, x):
        return self.weights(torch.nn.functional.relu(self.features(x)))


# Random features model with Poly activation
class RandomFeaturesPoly(nn.Module):
    def __init__(self, input_dimension, deg=6):
        super(RandomFeaturesPoly, self).__init__()
        self.num_feature = 2 ** 13
        self.deg = deg
        self.features = nn.Linear(input_dimension, self.num_feature)
        torch.nn.init.normal_(self.features.bias, mean=0, std=1/np.sqrt(input_dimension))
        torch.nn.init.normal_(self.features.weight, mean=0, std=1/np.sqrt(input_dimension))
        for param in self.features.parameters():
            param.required_grad = False
        self.weights = nn.Linear(self.num_feature, 1, bias=False)
        self.weights.weight.data.fill_(0.0)
    
    def forward(self, x):
        return self.weights(torch.pow(1+self.features(x), self.deg) / 1000) # 9496 is the 10-th moment of N(1,1) normalizing the activation function.


# A mean-field model based on description found in https://proceedings.mlr.press/v178/abbe22a.html
class MeanField(nn.Module):
    def __init__(self, input_dimension):
        super(MeanField, self).__init__()
        self.input_dimension = input_dimension
        self.width = 2 ** 16
        self.layer1 = nn.Linear(input_dimension, self.width)
        self.layer2 = nn.Linear(self.width, 1)
        self._init_weights(self.layer1, 0.5)
        self._init_weights(self.layer2, 0.0)

    def _init_weights(self, layer, alpha):
        if isinstance(layer, nn.Linear):
            bounds = 1. / (layer.weight.size(1) ** alpha)
            print(bounds)
            torch.nn.init.uniform_(layer.weight, -bounds, bounds)
            torch.nn.init.uniform_(layer.bias, -bounds, bounds)

    def forward(self, x):
        return self.layer2(torch.nn.functional.relu(self.layer1(x))) / self.width


# The MLP model used in experiments
class MLP(nn.Module):
    def __init__(self, input_dimension, alpha_init=2.0):
        super(MLP, self).__init__()
        self.input_dimension = input_dimension
        self.seq = nn.Sequential(
            nn.Linear(input_dimension, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.alpha_init = alpha_init
        # self.seq.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            stdv = 1. / layer.weight.size(1) ** self.alpha_init
            torch.nn.init.uniform_(layer.weight, -stdv, stdv)
            torch.nn.init.uniform_(layer.bias, -stdv, stdv)

    def forward(self, x):
        return self.seq(x)


##### Not used for the experiments reported in the paper #####

# Implementation of Maximal Update Model as described in https://arxiv.org/abs/2011.14522
class MaximalUpdate(nn.Module):
    def __init__(self, input_dimension, width, depth):
        super(MaximalUpdate, self).__init__()
        self.width = width
        self.depth = depth
        self.dims = [input_dimension]
        for _ in range(self.depth - 1): self.dims.append(self.width)
        self.dims.append(1)
        self.layers = nn.ModuleList()
        for i in range(self.depth):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            self._init_weights(self.layers[-1], beta=0.5)
            # if i != self.depth - 1:
            #     self.layers.append(nn.ReLU())
        print(self.layers)

    def _init_weights(self, layer, beta):
        if isinstance(layer, nn.Linear):
            std = self.width ** (-1 * beta)
            torch.nn.init.normal_(layer.weight, mean=0, std=std)
            torch.nn.init.normal_(layer.bias, mean=0, std=std)
            print(std)

    def forward(self, x):
        for i in range(self.depth):
            x = self.layers[i](x)
            if i == 0:
                x = np.sqrt(self.width) * x
            elif i == self.depth - 1:
                x = x / np.sqrt(self.width)
            if i != self.depth - 1:
                x = torch.nn.functional.relu(x)
        return x



# Neural Net with NTK parametrization as described in https://arxiv.org/abs/2011.14522 
class NTK(nn.Module):
    def __init__(self, input_dimension, width, depth):
        super(NTK, self).__init__()
        self.width = width
        self.depth = depth
        self.dims = [input_dimension]
        for _ in range(self.depth - 1): self.dims.append(self.width)
        self.dims.append(1)
        self.layers = nn.ModuleList()
        for i in range(self.depth):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            self._init_weights(self.layers[-1], beta=0.0)
            # if i != self.depth - 1:
            #     self.layers.append(nn.ReLU())
        print(self.layers)

    def _init_weights(self, layer, beta):
        if isinstance(layer, nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0, std=self.width ** (-1 * beta))
            torch.nn.init.normal_(layer.bias, mean=0, std=self.width ** (-1 * beta))

    def forward(self, x):
        for i in range(self.depth):
            x = self.layers[i](x)
            if i > 0:
                x = x / np.sqrt(self.width)
            if i != self.depth - 1:
                x = torch.nn.functional.relu(x)
        return x

