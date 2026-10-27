import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import albumentations as album
import ai8x
import cv2
from PIL import Image
from glob import glob


class GTSRB(Dataset):

    labels = ['give_way', 'no_entry', 'unknown']
    label_to_id_map = {k: v for v, k in enumerate(labels)}
    label_to_folder_map = {'13': 'give_way', '17': 'no_entry', '15': 'unknown'}

    def __init__(self, root_dir, d_type, transform=None,
                 resize_size=(128, 128), augment_data=False):
        
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, 'GTSRB', d_type)

        self.__get_image_paths()

        self.album_transform = None
        if d_type == 'train': # and augment_data
            self.album_transform = album.Compose([
                album.GaussNoise(var_limit=(1.0, 20.0), p=0.25),
                album.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                album.ColorJitter(p=0.5),
                album.SmallestMaxSize(max_size=int(1.2*min(resize_size))),
                album.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                album.RandomCrop(height=resize_size[0], width=resize_size[1]),
                album.HorizontalFlip(p=0.5),
                album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))])
        if not augment_data or d_type == 'test':
            self.album_transform = album.Compose([
                album.SmallestMaxSize(max_size=int(1.2*min(resize_size))),
                album.CenterCrop(height=resize_size[0], width=resize_size[1]),
                album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))])

        self.transform = transform
    
    def __get_image_paths(self):
        self.data_list = []

        for label in self.labels:
            image_dir = os.path.join(self.data_dir, label)
            print(label)
            for file_name in sorted(os.listdir(image_dir)):
                file_path = os.path.join(image_dir, file_name)
                if os.path.isfile(file_path):
                    self.data_list.append((file_path, self.label_to_id_map[label]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label = torch.tensor(self.data_list[index][1], dtype=torch.int64)

        image_path = self.data_list[index][0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.album_transform:
            image = self.album_transform(image=image)["image"]

        if self.transform:
            image = self.transform(image)
        
        return image, label

def GTSRB_get_datasets(data, load_train=True, load_test=True):
    """
    Load the MNIST dataset.

    The original training dataset is split into training and validation sets (code is
    inspired by https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb).
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-128/128, +127/128]
    https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

    Data augmentation: 4 pixels are padded on each side, and a 32x32 crop is randomly sampled
    from the padded image or its horizontal flip.
    This is similar to [1] and some other work that use CIFAR10.

    [1] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply Supervised Nets.
    arXiv:1409.5185, 2014
    """
    (data_dir, args) = data

    transform = transforms.Compose([
        transforms.ToTensor(),
        ai8x.normalize(args=args),
    ])

    if load_train:
        train_dataset = GTSRB(root_dir=data_dir, d_type='train',
                                   transform=transform, augment_data=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = GTSRB(root_dir=data_dir, d_type='test', transform=transform)
    else:
        test_dataset = None
    
    return train_dataset, test_dataset

datasets = [
    {
        'name': 'GTSRB',
        'input': (3, 128, 128),
        'output': ('give_way','no_entry','unknown'),
        'loader': GTSRB_get_datasets,
    }
]
