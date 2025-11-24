import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.base_folder = base_folder
        self.train = train
        self.transforms = transforms

        # cache all data samples from a file
        self.file_cache = {}

        if train:
            self.data_files = [f"data_batch_{i}" for i in range(1,6)]
        else:
            self.data_files = ["test_batch"]
        
        self.sample_size_per_file = self._get_samples_per_file()
        self.total_samples_size = sum(self.sample_size_per_file)

        ### END YOUR SOLUTION
    def _get_samples_per_file(self):
        """get sample size of each file only"""
        samples = []
        for file in self.data_files:
            with open(os.path.join(self.base_folder,file),'rb') as f:
                dict_data = pickle.load(f,encoding='bytes')
                samples.append(len(dict_data[b'labels']))
        return samples
    
    def _load_file(self,file_idx):
        """load a file's data to cache"""
        if file_idx not in self.file_cache:
            file_path = os.path.join(self.base_folder,self.data_files[file_idx])
            with open(file_path,'rb') as f:
                dict_data = pickle.load(f,encoding='bytes')
                # scale to (batch,3,32,32) and scale to 0~1
                data = dict_data[b'data'].reshape(-1,3,32,32).astype(np.float32) / 255.0
                self.file_cache[file_idx] = (data,dict_data[b'labels'])
        return self.file_cache[file_idx]

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        # which file does this sample belongs to?
        file_idx = 0
        idx_bias = 0
        for i,count in enumerate(self.sample_size_per_file):
            if index < idx_bias + count:
                file_idx=i
                break
            idx_bias += count
        
        # which location is this data sample at file with idx=file_idx
        sample_idx = index - idx_bias
        # load data of this file
        data,labels = self._load_file(file_idx)
        image = data[sample_idx]
        label = labels[sample_idx]
        if self.transforms:
            image = self.apply_transforms(image)
        return image,label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.total_samples_size
        ### END YOUR SOLUTION
