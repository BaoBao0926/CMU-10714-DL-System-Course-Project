import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
      if self.transforms is not None:
          # apply the transforms
          for tform in self.transforms:
              x = tform(x)
      return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                            range(batch_size, len(dataset), batch_size))

    def __iter__(self):
      indices = np.arange(len(self.dataset))
      if self.shuffle:
          np.random.shuffle(indices)

      self.ordering = np.array_split(
                  indices, 
                  np.arange(self.batch_size, len(indices), self.batch_size)
                )
      self._batch_id = 0 
      return self


    def __next__(self):
      if self._batch_id >= len(self.ordering):
          raise StopIteration

      batch_indices = self.ordering[self._batch_id]
      self._batch_id += 1

      # get batch from dataset
      batch = [self.dataset[i] for i in batch_indices]

      # get (images, labels)
      if isinstance(batch[0], tuple) and len(batch[0]) == 2 :
          xs, ys = zip(*batch)
          return Tensor(np.stack(xs)), Tensor(np.stack(ys))
      elif isinstance(batch[0], tuple) and len(batch[0]) == 1:
        xs, = zip(*batch)   # for tuple with len of 1
        return (Tensor(np.stack(xs)),)
      else:
          return (Tensor(np.stack(batch)),)
















