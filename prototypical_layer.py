# coding=utf-8
import torch
from torch.nn import functional as F

# inspired by prototypical_loss at https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
from utils import euclidean_dist


class PrototypicalLayer(torch.nn.Module):
    def __init__(self, classes_per_iter):
        super(PrototypicalLayer, self).__init__()
        self.N = classes_per_iter

    def forward(self, input):
        '''
        Args:
        - input: the model output for a batch of samples, where first N rows are support vectors
        '''

        prototypes = input[:self.N] # N
        # our version uses all tensors as query
        query_samples = input # M
        dists = euclidean_dist(query_samples, prototypes) # MxN
        log_p_y = F.log_softmax(-dists, dim=1) # MxN
        return log_p_y
