import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, size_list, activation="relu", batch_norm=True):
        super(MLP, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features=size_list[i+1]))
            if activation is "relu":
                layers.append(nn.ReLU())
            else:
                raise NotImplementedError
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)