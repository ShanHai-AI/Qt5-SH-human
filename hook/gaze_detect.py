import argparse
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
from PIL import Image
from lazernet.utils import select_device, draw_gaze
from PIL import Image, ImageOps
# from face_detection import RetinaFace
from lazernet.model import L2CS

gpu_id="0"
gpu = select_device(gpu_id, batch_size=1)

def init_model(arch='ResNet50',snapshot_path="lazernet/models/L2CSNet/Gaze360/L2CSNet_gaze360.pkl"):
    cudnn.enabled = True
    model=getArch(arch, 90)
    # print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()
    return model


def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model


def gaze_estimate(model,frame,faces):
    """
    Args:
        model: init_model return model
        img: face
    Returns:
    """
    # 数据处理
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    softmax = nn.Softmax(dim=1)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x=0

    if faces is not None:
        for box, track_id, score in faces:
            if score < .5: #.95
                continue
            x_min=int(box[0])
            if x_min < 0:
                x_min = 0
            y_min=int(box[1])
            if y_min < 0:
                y_min = 0
            x_max=int(box[2])
            y_max=int(box[3])
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            # x_min = max(0,x_min-int(0.2*bbox_height))
            # y_min = max(0,y_min-int(0.2*bbox_width))
            # x_max = x_max+int(0.2*bbox_height)
            # y_max = y_max+int(0.2*bbox_width)
            # bbox_width = x_max - x_min
            # bbox_height = y_max - y_min
            # Crop image
            img = frame[y_min:y_max, x_min:x_max]
            img = cv2.resize(img, (224, 224))
            # cv2.imshow("img",img)
            # cv2.waitKey()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            img=transformations(im_pil)
            img  = Variable(img).cuda(gpu)
            img  = img.unsqueeze(0)
            # print(img,"*****")
            # gaze prediction
            try:
                gaze_pitch, gaze_yaw = model(img)
                pitch_predicted = softmax(gaze_pitch)
                yaw_predicted = softmax(gaze_yaw)
                # Get continuous predictions in degrees.
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

                draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch_predicted,yaw_predicted),color=(0,0,255))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,255,0), 1)
            except:
                pass
    return frame

