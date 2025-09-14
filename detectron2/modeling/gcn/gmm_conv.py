import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from .inits import reset, glorot, zeros

EPS = 1e-15


class GMMConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dim,
                 kernel_size,
                 bias=True,
                 **kwargs):
        super(GMMConv, self).__init__(aggr='add', **kwargs)

        

