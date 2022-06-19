# coding=utf-8
import numpy as np
import torch

#  inspired by PrototypicalBatchSampler at https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, num_support, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations
        self.num_support = num_support

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        print(f"{len(self.classes)} classes")
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            # batch_size = spc * cpi
            # batch = torch.LongTensor(batch_size)
            support_indices = []
            query_indices = []
            c_idxs, _ = torch.sort(torch.randperm(len(self.classes))[:cpi])
            for i, c in enumerate(self.classes[c_idxs]): # self.classes is sorted, hence c is sorted too
                # s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                # batch[s] = self.indexes[label_idx][sample_idxs]
                support_indices.append(self.indexes[label_idx][sample_idxs[:self.num_support]])
                query_indices.append(self.indexes[label_idx][sample_idxs[self.num_support:]])
            indices = support_indices + query_indices # we put the support indices at the beginning of batch
            batch = torch.cat(indices).long()
            # batch = batch[torch.randperm(len(batch))]
            # different from the original version, the beginning of each batch are support indices of length cpi
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
