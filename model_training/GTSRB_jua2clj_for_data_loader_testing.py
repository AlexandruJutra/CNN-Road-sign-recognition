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
from types import SimpleNamespace

class GTSRB(Dataset):

    labels = ['give_way', 'no_entry', 'priority_road', 'stop']
    label_to_id_map = {k: v for v, k in enumerate(labels)}
    label_to_folder_map = {'0': 'give_way', '1': 'no_entry', '2': 'priority_road', '3': 'stop'}

    def __init__(self, root_dir, d_type, transform=None,
                 resize_size=(32, 32), augment_data=False):
        
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, 'GTSRB', d_type)

        self.__get_image_paths()

        self.album_transform = None
        if d_type == 'train' and augment_data:
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
        
        print(f"Loaded image path: {image_path}, label: {label}")

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

    x = train_dataset.__len__()
    for i in range(x):
        train_dataset.__getitem__(i)
    print('jua2clj@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    return train_dataset, test_dataset

datasets = [
    {
        'name': 'GTSRB',
        'input': (3, 32, 32),
        'output': ('priority_road', 'give_way', 'stop', 'no_entry'),
        'loader': GTSRB_get_datasets,
    }
]


def main():
    args = SimpleNamespace(
    device='cpu',
    act_mode_8bit=False,
    cnn='ai85simplenet',
    out_fold_ratio=1,
    dataset='GTSRB',
    truncate_testset=False,
    data='data',
    workers=4,
    epochs=300,
    batch_size=100,
    use_bias=True,
    avg_pool_rounding=False,
    dr=None,
    scaf_margin=28.6,
    scaf_scale=64,
    backbone_checkpoint=None,
    copy_output_folder=None,
    kd_relationbased=False,
    qat_policy='policies/qat_policy_cifar100.yaml',
    nas=False,
    nas_kd_resume_from='',
    nas_policy='',
    nas_stage_transition_list=None,
    optimizer='Adam',
    lr=0.001,
    scaf_lr=0.0001,
    momentum=0.9,
    weight_decay=0.0,
    print_freq=100,
    resumed_checkpoint_path='',
    load_model_path='',
    pretrained=False,
    reset_optimizer=False,
    evaluate=False,
    generate_sample=None,
    slice_sample=False,
    activation_stats=[],
    log_params_histograms=False,
    summary=None,
    summary_filename='model',
    yaml_template=None,
    compress='policies/schedule-cifar100.yaml',
    deterministic=False,
    seed=None,
    gpus=-1,
    cpu=False,
    name=None,
    output_dir='logs',
    validation_split=0.0,
    display_confusion=False,
    display_embedding=False,
    display_prcurves=False,
    show_train_accuracy='last_batch',
    no_tensorboard=True,
    tblog=False,
    regression=False,
    num_best_scores=1,
    compiler_mode='default',
    compiler_backend='inductor',
    obj_detection=False,
    obj_detection_params=None,
    kd_teacher=None,
    kd_pretrained=False,
    kd_resume='',
    kd_temp=1.0,
    kd_distill_wt=0.5,
    kd_student_wt=0.5,
    kd_teacher_wt=0.0,
    kd_start_epoch=0,
    local_world_size=1,
    labels=('priority_road', 'give_way', 'stop', 'no_entry'),
    num_classes=4,
    dimensions=(3, 32, 32),
    # datasets_fn=<function GTSRB_get_datasets at 0x7c5c3fd609a0>,
    collate_fn=None,
    # visualize_fn=<function visualize_data at 0x7c5c404fd800>
    )
    data_path = 'data'
    data = data_path, args
    GTSRB_get_datasets(data)

if __name__ == "__main__":
    main()
