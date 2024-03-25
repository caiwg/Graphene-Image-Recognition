import cv2
import numpy as np
# from headers import exposure
from PIL import Image
from PIL import ImageEnhance


def image_preprocessing(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    img_smoothed = cv2.GaussianBlur(img, (5, 5), 0)
    img_result = cv2.Canny(img_smoothed, 50, 100)

    ret, binary = cv2.threshold(img_result, 160, 255, cv2.THRESH_BINARY)  # 图片二值化,灰度值大于40赋值255，反之0

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    threshold = 150  # 设定阈值
    # cv2.fingContours寻找图片轮廓信息
    """提取二值化后图片中的轮廓信息 ，返回值contours存储的即是图片中的轮廓信息，是一个向量，内每个元素保存
    了一组由连续的Point点构成的点的集合的向量，每一组Point点集就是一个轮廓，有多少轮廓，向量contours就有
    多少元素"""
    contours, hierarch = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])  # 计算轮廓所占面积
        if area < threshold:  # 将area小于阈值区域填充背景色，由于OpenCV读出的是BGR值
            cv2.drawContours(img, [contours[i]], -1, (176, 75, 247), thickness=-1)  # 原始图片背景BGR值(84,1,68)
            continue

    return img


def ImageAugument(imgPath):
    image = Image.open(imgPath)

    # 色度增强
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    image_colored = enh_col.enhance(color)
    image_colored.save('/Users/cws/Desktop/论文/Unet/image/6_2.jpg')

    # 对比度增强
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    image_contrasted.save('/Users/cws/Desktop/论文/Unet/image/6_2.jpg')
    #
    # # 锐度增强
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = 3.0
    image_sharped = enh_sha.enhance(sharpness)
    image_sharped.save('/Users/cws/Desktop/论文/Unet/image/5_2.jpg')

# 读取图像并进行预处理
image_path = '/Users/cws/Desktop/论文/Unet/image/ori/5.jpg'
processed_image = image_preprocessing(image_path)

# 保存处理后的图像
output_path = '/Users/cws/Desktop/论文/Unet/image/5_3.jpg'
cv2.imwrite(output_path, processed_image)
ImageAugument(output_path)
print(f"Processed image saved at: {output_path}")
