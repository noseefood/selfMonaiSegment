
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
    RandGaussianSmooth,
    AdjustContrast,
    RandFlip,   
    RandAxisFlip,
    

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
epoch_num = 40
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

    train_imtrans = Compose(
        [
            # basic_transform,
            LoadImage(image_only=True, ensure_channel_first=True),
            Resize((512, 512)), # 必须要加入这个，否则会报错，这里相当于直接拉伸
            # RandGaussianNoise(prob=0.3),  
            # RandGaussianSmooth(prob=0.3),
            # ScaleIntensity(), # 0-1 注意顺序 这个顺序下似乎学不到东西有点奇怪
            ScaleIntensity(),  
            RandGaussianNoise(prob=0.3),
            RandGaussianSmooth(prob=0.3),
            # RandAxisFlip(prob=0.7), # 两个方向都有可能翻转，会在ArrayDataset里面自动同步
            RandFlip(prob=0.7), # 两个方向都有可能翻转，会在ArrayDataset里面自动同步
        ]
    )
    train_segtrans = Compose(
        [
            # basic_transform,
            LoadImage(image_only=True, ensure_channel_first=True),
            Resize((512, 512)), 
            ScaleIntensity(),
            # RandAxisFlip(prob=0.7),
            RandFlip(prob=0.7),
        ]
    )


    # val_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), Resize((512, 512)), ScaleIntensity()])   # 理论上validation不需要加入额外的变换
    # val_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), Resize((512, 512)), ScaleIntensity()])

 
    val_imtrans = Compose(
        [
            # basic_transform,
            LoadImage(image_only=True, ensure_channel_first=True),
            Resize((512, 512)), # 必须要加入这个，否则会报错，这里相当于直接拉伸
            # RandGaussianNoise(prob=0.3),
            # RandGaussianSmooth(prob=0.3),
            # ScaleIntensity(),
            ScaleIntensity(),
            RandGaussianNoise(prob=0.3),
            RandGaussianSmooth(prob=0.3),
            # RandAxisFlip(prob=0.7), # 两个方向都有可能翻转，会在ArrayDataset里面自动同步
            RandFlip(prob=0.7), # 两个方向都有可能翻转，会在ArrayDataset里面自动同步
        ]
    )
    val_segtrans = Compose(
        [
            # basic_transform,
            LoadImage(image_only=True, ensure_channel_first=True),
            Resize((512, 512)), 
            ScaleIntensity(),
            # RandAxisFlip(prob=0.7),
            RandFlip(prob=0.7),
        ]
    )   

    test_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), Resize((512, 512)), ScaleIntensity()])   # 理论上test不需要加入额外的变换
    test_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), Resize((512, 512)), ScaleIntensity()])


    # create data loader and split the dataset
    img_train_val, img_test, seg_train_val, seg_test = train_test_split(images, segs, test_size=0.10, random_state=42)  # shuffle=True default
    img_train, img_val, seg_train, seg_val = train_test_split(img_train_val, seg_train_val, test_size=val_percent, random_state=42)

    # # 注意ArrayDataset会自动让两个变换的seed保持一致！详见https://github.com/Project-MONAI/MONAI/discussions/2983
    train_ds = ArrayDataset(img_train, train_imtrans, seg_train, train_segtrans)  
    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
    val_ds = ArrayDataset(img_val, val_imtrans, seg_val, val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=batch_size_val, num_workers=4, pin_memory=torch.cuda.is_available()) # 相当于每次val只有一张图片

    test_ds = ArrayDataset(img_test, test_imtrans, seg_test, test_segtrans)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available()) # 相当于每次val只有一张图片


    print(f"train dataset size: {len(train_ds)}, val dataset size: {len(val_ds)}")
    print(f"test dataset size: {len(test_ds)}")
    
    # image original shape TODO 训练的时候其实不太需要
    


    # DICE metric
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])  # !!!绝对不要忘了，决定了最后的mask输出

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),   
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True) # DICE loss
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter() 

    for epoch in range(epoch_num):  
        print("-" * 10)
        print(f"epoch {epoch + 1}/{10}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device) # torch.Size([8, 1, 512, 512])

            inputs, labels = inputs.transpose(-1,-2), labels.transpose(-1,-2)  # 在推理的时候也要注意这个

            # print("inputs's shape", inputs.shape) # torch.Size([8, 1, 512, 512])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step # epoch_loss对每一步求平均
        epoch_loss_values.append(epoch_loss)
        # print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # validation 
        if (epoch + 1) % val_interval == 0: # 每隔val_interval个epoch进行一次validation
            model.eval()
            with torch.no_grad():       
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader: # 

                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)

                    val_images, val_labels = val_images.transpose(-1,-2), val_labels.transpose(-1,-2) 

                    # print("val_images's shape", val_images.shape) # val_images's shape torch.Size([1, 1, 512, 512]) ！！！DataLoader注意会增加一个维度详见推理
                    val_outputs = model(val_images)
                    dice_metric(y_pred=post_trans(val_outputs), y=val_labels)

                # aggregate the final mean dice result for all val_data of val_loader
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_segmentation2d_array.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)

                # plot the last model output as GIF image in TensorBoard with the corresponding image and label(only in TensorBoard)
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="Original image")  # 注意only plot the first in the batch
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="Original label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="Model output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    



    # final test using test_loader
    model.eval()
    with torch.no_grad():  
        test_images = None
        test_labels = None
        test_outputs = None
        test_metric = -1
        for test_data in test_loader: # 

            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)

            test_images, test_labels = test_images.transpose(-1,-2), test_labels.transpose(-1,-2) 

            # print("val_images's shape", test_images.shape) # val_images's shape torch.Size([1, 1, 512, 512]) ！！！DataLoader注意会增加一个维度详见推理
            test_outputs = model(test_images)
            dice_metric(y_pred=post_trans(test_outputs), y=test_labels)
            # print(dice_metric)

        # aggregate the final mean dice result for all test_data of test_loader 自动求平均
        test_metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

        writer.add_scalar("final test_mean_dice", metric)
        print("final test_mean_dice: {:.4f}".format(test_metric))

    writer.close()



if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)

















