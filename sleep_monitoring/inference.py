from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
import torch
import numpy as np
from nets import yolo3
yolo = YOLO()
import cv2
torch.backends.cudnn.benchmark = True
from PIL import Image, ImageDraw, ImageFont


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def paint_chinese_opencv(im,chinese,pos,color):
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('NotoSansCJK-Bold.ttc',25)
    fillColor = color #(255,0,0)
    position = pos #(100,100)
    if not isinstance(chinese,unicode):
        chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,font=font,fill=fillColor)
 
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    return img


def cv2ImgAddText(img, text, left, top, textColor=(192, 192, 192), textSize=40):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # print(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "/home/zhangyn/下载/睡眠inference/NotoSansCJK-DemiLight.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# content=None

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device=torch.device('cpu')
    print("Device being used:", device)

    with open('./model_data/new_classes.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    # init model
    model = yolo3.YoloBody(num_classes=3,anchor=1)
    checkpoint = torch.load('model_data/Epoch95-Total_Loss1.8304-Val_Loss1.5908.pth', map_location=lambda storage, loc: storage)
    #state_dict = torch.load('model_data/Epoch95-Total_Loss1.8304-Val_Loss1.5908.pth', map_location=device)
    
    """
    state_dict = model.state_dict()
    for k1, k2 in zip(state_dict.keys(), checkpoint.keys()):
        state_dict[k1] = checkpoint[k2]
    model.load_state_dict(state_dict)
    """
    model.load_state_dict(checkpoint['state_dict'])#模型参数
    #optimizer.load_state_dict(checkpoint['opt_dict'])#优化参数
    
    model.to(device)
    model.eval()

    # read video
    #video = './data/UCF-101/normal/v_normal_g01_c01.avi'
    #video = '1.mp4'
    #cap = cv2.VideoCapture(video)
    # 调用摄像头识别动作
    cap = cv2.VideoCapture(0)
    retaining = True
    #运行后保存的文件名
    result_video = "demoresult.mp4"
    #读取视频
    cap = cv2.VideoCapture(video)
    #获取视频帧率
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    #设置写入视频的编码格式
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #获取视频宽度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #获取视频高度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ####重要
    videoWriter = cv2.VideoWriter(result_video, fourcc, fps_video,(frame_width,frame_height))

    clip = []
    while retaining:
        retaining, framec = cap.read()
        frame2=framec
        # """
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(framec,cv2.COLOR_BGR2RGB)
        # 转变成Image
        frameX= Image.fromarray(np.uint8(frame))

        # 进行检测
        frameXXX,label_ = np.array(yolo.detect_image(frameX))
        # print(label)
        if float(str(label_).split(' ')[1].strip('\''))>=0.75:
            frame=frame2
            tmp_ = center_crop(cv2.resize(frame, (171, 128)))
            tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
            clip.append(tmp)
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
                inputs = torch.from_numpy(inputs)
                inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
                with torch.no_grad():
                    outputs = model.forward(inputs)

                probs = torch.nn.Softmax(dim=1)(outputs)
                label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]


                if class_names[label].split(' ')[-1].strip()=='1':
                    content='平躺'
                elif class_names[label].split(' ')[-1].strip()=='2':
                    content='翻身'
                else:
                    content='坐起'

                img = cv2ImgAddText(frame2, content, 60, 60)
                frame=img
                # cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (60, 40),
                #            cv2.FONT_HERSHEY_SIMPLEX, 2,
                #            (0, 0, 255), 10)
                # cv2.putText(frame, "prob: %.4f" % probs[0][label], (60, 90),
                #            cv2.FONT_HERSHEY_SIMPLEX, 2,
                #            (0, 0, 255), 10)
                clip.pop(0)
        else:
             cv2.putText(frame2, 'None', (60, 40),cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255), 10)
             frame=frame2
        videoWriter.write(frame)
        cv2.imshow('result', frame)
        cv2.waitKey(30)

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()









