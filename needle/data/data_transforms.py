import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C NDArray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
          if img.ndim == 2: # (H,W)
            return img[:, ::-1]  # H W， horizontal->W
          if img.ndim == 3:
            return img[:, ::-1, :]  # H W C， horizontal->W
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """
        Zero pad and then randomly crop an image.
        """
        if img.ndim == 2:  # (H, W)
            H, W = img.shape
            padded_img = np.pad(
                img,
                ((self.padding, self.padding), (self.padding, self.padding)),
                mode="constant"
            )

            shift_x, shift_y = np.random.randint(-self.padding, self.padding + 1, size=2)
            start_x = self.padding + shift_x
            start_y = self.padding + shift_y

            cropped_img = padded_img[start_x:start_x + H, start_y:start_y + W]

        elif img.ndim == 3:  # (H, W, C)
            H, W, C = img.shape
            padded_img = np.pad(
                img,
                ((self.padding, self.padding),
                 (self.padding, self.padding),
                 (0, 0)), 
                mode="constant"
            )

            shift_x, shift_y = np.random.randint(-self.padding, self.padding + 1, size=2)
            start_x = self.padding + shift_x
            start_y = self.padding + shift_y

            cropped_img = padded_img[start_x:start_x + H, start_y:start_y + W, :]

        else:
            raise ValueError(f"Unsupported image shape {img.shape}")

        return cropped_img
