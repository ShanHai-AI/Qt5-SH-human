# 睡眠监测
#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from PIL import Image, ImageDraw, ImageFont
from sleep_monitoring.yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time

# 模型初始化
yolo = YOLO()

def cv2ImgAddText(img, text, left, top, textColor=(192, 192, 192), textSize=40):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # print(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
         "/home/cwh/桌面/SH-human-main/sleep_monitoring/model_data/simhei.ttf", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def sleep_detect(frame):
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame= Image.fromarray(np.uint8(frame))
    image,label=yolo.detect_image(frame)
    frame = np.array(image)
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    if label.decode().split(' ')[0].strip() == '1':
        # print('cxcccccccccccc', label.decode().split(' ')[0].strip())
        content = 'lie down'
    elif label.decode().split(' ')[0].strip() == '2':
        content = 'turn over'
    elif label.decode().split(' ')[0].strip() == '3':
        content = 'sit up'
    else:
    	content = 'unknown'
    # frame = cv2ImgAddText(frame, content, 60, 60, textColor=( 255,0, 0))
    return frame,content
