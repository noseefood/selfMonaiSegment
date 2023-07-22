
import logging
import os
import sys
import tempfile
from glob import glob

import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import monai
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    DivisiblePadd,
    Resize,
    RandGaussianNoise,
    AdjustContrast,
    Flip,
    

)
from monai.visualize import plot_2d_or_3d_image
from torch.utils.data import random_split


from pathlib import Path
base = Path(os.environ['raw_data_base']) if 'raw_data_base' in os.environ.keys() else Path('./data')
assert base is not None, "Please assign the raw_data_base(which store the training data) in system path "
dir_img = base / 'imgs'
dir_mask = base / 'masks/'
dir_checkpoint = 'checkpoints/'

####### train parameters #######
val_percent = 0.25  # train dataset size: 1710, val dataset size: 571 test dataset size: 254
epoch_num = 25
batch_size_train = 8
batch_size_val = 1
#################################


def main(tempdir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # load raw data
    images = sorted(glob(os.path.join(dir_img, "*.png"))) # all in list
    segs = sorted(glob(os.path.join(dir_mask, "*.png")))
    if len(images) == len(segs): # 二者顺序是否能对的上还需要判断
        print("imgs and masks lists are same...")
    else:
        assert False, "imgs and masks lists are not same..."


    # define transforms for image and segmentation

    basic_transform = Compose(
            LoadImage(image_only=True, ensure_channel_first=True),
            Resize((512, 512)), # 必须要加入这个，否则会报错，这里相当于直接拉伸
            ScaleIntensity(),    
    )

    train_imtrans = Compose(
        [
            # LoadImage(image_only=True, ensure_channel_first=True),
            # Resize((512, 512)), # 必须要加入这个，否则会报错，这里相当于直接拉伸
            # ScaleIntensity(),
            basic_transform,


            RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),
            # AdjustContrast(gamma=0.5),
            


        ]
    )
    train_segtrans = Compose(
        [
            # LoadImage(image_only=True, ensure_channel_first=True),
            # Resize((512, 512)), 
            # ScaleIntensity(),
            basic_transform,
        ]
    )

    keys = ["image", "label"]
    train_transform = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True, keys=keys),
            Resize((512, 512), keys=keys), 
            ScaleIntensity(keys=keys),
        ]
    )

    
    val_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), Resize((512, 512)), ScaleIntensity()])   # 理论上validation不需要加入额外的变换
    val_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), Resize((512, 512)), ScaleIntensity()])

    test_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), Resize((512, 512)), ScaleIntensity()])   # 理论上test不需要加入额外的变换
    test_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), Resize((512, 512)), ScaleIntensity()])


    # create data loader and split the dataset
    img_train_val, img_test, seg_train_val, seg_test = train_test_split(images, segs, test_size=0.10, random_state=42)  # shuffle=True default
    img_train, img_val, seg_train, seg_val = train_test_split(img_train_val, seg_train_val, test_size=val_percent, random_state=42)

    train_ds = ArrayDataset(img_train, train_imtrans, seg_train, train_segtrans)  # 注意ArrayDataset会自动让两个变换的seed保持一致！详见https://github.com/Project-MONAI/MONAI/discussions/2983
    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
    val_ds = ArrayDataset(img_val, val_imtrans, seg_val, val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=batch_size_val, num_workers=4, pin_memory=torch.cuda.is_available()) # 相当于每次val只有一张图片

    test_ds = ArrayDataset(img_test, test_imtrans, seg_test, test_segtrans)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available()) # 相当于每次val只有一张图片


    print(f"train dataset size: {len(train_ds)}, val dataset size: {len(val_ds)}")
    print(f"test dataset size: {len(test_ds)}")

    
    # image original shape TODO 训练的时候其实不太需要
    


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)

















