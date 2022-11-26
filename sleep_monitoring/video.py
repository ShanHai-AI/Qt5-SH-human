#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from PIL import Image, ImageDraw, ImageFont
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time

def cv2ImgAddText(img, text, left, top, textColor=(192, 192, 192), textSize=40):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # print(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
         "model_data/simhei.ttf", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

yolo = YOLO()
# 调用摄像头
capture=cv2.VideoCapture(0,cv2.CAP_DSHOW)
#capture=cv2.VideoCapture(0) # capture=cv2.VideoCapture("1.mp4")

fps = 0.0
while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    # 转变成Image
    frame= Image.fromarray(np.uint8(frame))
    #cv2.imshow('1',frame)
    #cv2.waitKey(0)
    image,label=yolo.detect_image(frame)
    result_video = "demoresult.mp4"
    fps_video = capture.get(cv2.CAP_PROP_FPS)
    # 设置写入视频的编码格式
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # 获取视频宽度
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 获取视频高度
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ####重要
    videoWriter = cv2.VideoWriter(result_video, fourcc, fps_video, (frame_width, frame_height))
    # 进行检测
    frame = np.array(image)

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    if label.decode().split(' ')[0].strip() == '1':
        print('cxcccccccccccc', label.decode().split(' ')[0].strip())
        content = '平躺'
    elif label.decode().split(' ')[0].strip() == '2':
        content = '翻身'
    elif label.decode().split(' ')[0].strip() == '3':
        content = '坐起'
    else:
    	  content = '未知'
    frame = cv2ImgAddText(frame, content, 60, 60, textColor=( 255,0, 0))
    # fps  = ( fps + (1./(time.time()-t1)) ) / 2
    # print("fps= %.2f"%(fps))
    # frame = cv2.putText(frame, label, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    videoWriter.write(frame)
    cv2.namedWindow('video',cv2.WND_PROP_FULLSCREEN)
    cv2.imshow("video",frame)


    c= cv2.waitKey(1) & 0xff 
    if c==27:
        capture.release()
        break
