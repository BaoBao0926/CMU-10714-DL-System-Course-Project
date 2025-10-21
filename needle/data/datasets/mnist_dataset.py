from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct, gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
      # get images
      with gzip.open(image_filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols) # [B H W]
      # get labels
      with gzip.open(label_filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        self.labels = np.frombuffer(f.read(), dtype=np.uint8)
      self.transforms = transforms


    def __getitem__(self, index) -> object:
      image = self.images[index].astype(np.float32)/255.0
      image = self.apply_transforms(image)
      label = self.labels[index]
      return image, label

    def __len__(self) -> int:
      return len(self.images)

