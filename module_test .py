import numpy as np
import monai.transforms as T
import cv2

# 读取一张图像（NumPy格式）
image = np.random.rand(128, 128)

# 定义变换操作
transform = T.Compose([
    T.AddChannel(),  # 增加通道维度
    T.ScaleIntensity(),  # 缩放像素值到 [0, 1] 范围
    T.RandRotate90(prob=1.0, spatial_axes=(0, 1)),  # 随机旋转90度
])

# 对图像进行变换
transformed_image = transform(image)


# transformed_image 现在是处理后的图像，是一个形状为 (1, 128, 128) 的 NumPy 数组
print(transformed_image.shape)