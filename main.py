from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon

from myutils.CustomMessageBox import MessageBox
from myutils.capnums import Camera
from dialog.rtsp_win import Window


import sys
import os
import json
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2

## yolov5
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
# from yolov5.utils.datasets import LoadImages, LoadWebcam
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from stgcn.ActionsEstLoader import TSSTG
import hook.skp_pose1 as skp


class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # 发送信号：正在检测/暂停/停止/检测结束/错误报告
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov5n.engine'           # 设置权重
        self.current_weight = './yolov5n.engine'    # 当前权重
        self.source = '/home/cwh/桌面/SH-human-main/testvideo/fall_01.avi'                       # 视频源
        self.conf_thres = 0.25                  # 置信度
        self.iou_thres = 0.45                   # iou
        self.jump_out = False                   # 跳出循环
        self.is_continue = True                 # 继续/暂停
        self.percent_length = 1000              # 进度条
        self.rate_check = True                  # 是否启用延时
        self.rate = 100                         # 延时HZ
        self.save_fold = './result'             # 保存文件夹

    def run(self,
            yolo_weights=WEIGHTS / 'yolov5n.engine',  # model.pt path(s),
            strong_sort_weights=WEIGHTS / 'osnet_x0_5_msmt17.engine',  # model.pt path,
            config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            show_vid=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            save_vid=False,  # save confidences in --save-txt labels
            nosave=False,  # do not save images/videos
            classes=0,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/track',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=2,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            hide_class=False,  # hide IDs
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            eval=False,  # run multi-gpu eval
                 ):
        source = str(self.source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        if not isinstance(yolo_weights, list):  # single yolo model
            exp_name = yolo_weights.stem
        elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
            exp_name = Path(yolo_weights[0]).stem
        else:  # multiple models after --yolo_weights
            exp_name = 'ensemble'
        exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
        save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
        (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        if eval:
            device = torch.device(int(device))
        else:
            device = select_device(device)
        model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        print("i")
        # dataloader初始化
        if webcam:
            show_vid = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride,auto=pt)
            nr_sources = len(dataset)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride,auto=pt)
            nr_sources = 1
        vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

        # initialize StrongSORT
        cfg = get_config()
        cfg.merge_from_file(config_strongsort)
        # print(cfg)

        # Create as many strong sort instances as there are video sources
        strongsort_list = []
        for i in range(nr_sources):
            strongsort_list.append(
                StrongSORT(
                    strong_sort_weights,
                    device,
                    half,
                    max_dist=cfg.STRONGSORT.MAX_DIST,
                    max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.STRONGSORT.MAX_AGE,
                    n_init=cfg.STRONGSORT.N_INIT,
                    nn_budget=cfg.STRONGSORT.NN_BUDGET,
                    mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                    ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

                )
            )
            strongsort_list[i].model.warmup()
        outputs = [None] * nr_sources

        # print(strongsort_list)
        # Run tracking
        model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
        count = 0
        # 跳帧检测
        jump_count = 0
        start_time = time.time()
        dataset = iter(dataset)
        # 模型初始化
        action_model = TSSTG()
        # 1123 变量定义
        multistream_results = []
        for ii in range(nr_sources):
            multistream_results.append({})

        while True:
            # 手动停止
            if self.jump_out:
                self.vid_cap.release()
                self.send_percent.emit(0)
                self.send_msg.emit('停止')
                if hasattr(self, 'out'):
                    self.out.release()
                break
            # 临时更换模型
            if self.current_weight != self.weights:
                # Load model
                model = DetectMultiBackend(self.weights, device=device, dnn=dnn, data=None, fp16=half)
                stride, names, pt = model.stride, model.names, model.pt
                model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
                self.current_weight = self.weights
            # 暂停开关
            if self.is_continue:
                path, im, im0s, self.vid_cap, s = next(dataset)
                # jump_count += 1
                # if jump_count % 5 != 0:
                #     continue
                count += 1
                # 每三十帧刷新一次输出帧率
                if count % 30 == 0 and count >= 30:
                    fps = int(30/(time.time()-start_time))
                    self.send_fps.emit('fps：'+str(fps))
                    start_time = time.time()
                if self.vid_cap:
                    percent = int(count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length)
                    self.send_percent.emit(percent)
                else:
                    percent = self.percent_length
                statistic_dic = {name: 0 for name in names}
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                # Inference
                visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()


                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3

                # Process detections
                for i, det in enumerate(pred):  # detections per image

                    # 1123 多流的每一个流
                    # multistream_results[i]= dict()

                    seen += 1
                    if webcam:  # nr_sources >= 1
                        p, im0, _ = path[i], im0s[i].copy(), dataset.count
                        p = Path(p)  # to Path
                        s += f'{i}: '
                        txt_file_name = p.name
                        save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                    else:
                        p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                        p = Path(p)  # to Path
                        # video file
                        if source.endswith(VID_FORMATS):
                            txt_file_name = p.stem
                            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                        # folder with imgs
                        else:
                            txt_file_name = p.parent.name  # get folder name containing current img
                            save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
                    curr_frames[i] = im0

                    txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    imc = im0.copy() if save_crop else im0  # for save_crop

                    annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
                    if cfg.STRONGSORT.ECC:  # camera motion compensation
                        strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        xywhs = xyxy2xywh(det[:, 0:4])
                        confs = det[:, 4]
                        clss = det[:, 5]
                        # print("xywhs",xywhs.cpu())
                        # print(confs)
                        # print(clss)

                        # pass detections to strongsort
                        t4 = time_sync()
                        outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                        t5 = time_sync()
                        dt[3] += t5 - t4

                        # print("outputs[i]",outputs[i])
                        #----------------------------------------------#
                        ################################################
                        # 其他算法逻辑-HOOK
                        ################################################
                        if "pose"=="pose":
                            # 1123
                            pose_results, vis_img, pose_results_dict=skp.skeleton_detect(outputs[i],imc)
                            print(pose_results_dict,"pose_results_dict")
                            annotator.im=vis_img
                            # cv2.imshow('Image', vis_img)
                            # cv2.waitKey(0)


                        # draw boxes for visualization
                        if len(outputs[i]) > 0:
                            for j, (output, conf) in enumerate(zip(outputs[i], confs)):

                                bboxes = output[0:4]
                                id = int(output[4])
                                cls = output[5]


                                # 1123 每一个跟踪结果
                                # print(multistream_results)
                                # ---------------- 目标未出现过 初始化其信息 -----------------
                                if id not in multistream_results[i]:
                                    # print("init")
                                    # -------------------------------------------------------------------message_dict
                                    # message_dict[id] = (位置信息, 是否过线, [是否终结本目标的OCR，结果1，结果2...], 颜色, 姓名)
                                    # message_dict[id] = [bboxes, False, ["False", ""], "color", "unknown"]
                                    multistream_results[i][id] = {
                                        "skp_step": [],
                                        "fall_flag":False,
                                    }
                                    print(multistream_results)

                                else:
                                    # ----------------目标出现过--------------------
                                    # ---------------------------------------------
                                    #                  判断跌倒
                                    # ---------------------------------------------
                                    print(len(multistream_results[i][id]["skp_step"]))
                                    if len(multistream_results[i][id]["skp_step"]) < 30:
                                        try:
                                            person_skp = pose_results_dict[id]["keypoints"][:13]
                                            multistream_results[i][id]["skp_step"].append(person_skp)
                                        except:
                                            pass
                                    else:
                                        print("====================== action prediction")
                                        pts = np.array(multistream_results[i][id]["skp_step"], dtype=np.float32)
                                        out = action_model.predict(pts, vis_img.shape[:2])
                                        action_name = action_model.class_names[out[0].argmax()]

                                        if action_name == 'Fall Down' or action_name == 'Lying Down':
                                            print("跌倒")
                                            multistream_results[i][id]["fall_flag"] = True
                                        else:
                                            multistream_results[i][id]["fall_flag"] = False

                                        multistream_results[i][id]["skp_step"].clear()

                                    if multistream_results[i][id]["fall_flag"]:
                                        cv2.putText(vis_img, 'FALL', (int(bboxes[0]), int(bboxes[1])+50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                                    # ------------------跌倒判断结束--------------------

                                if save_txt:
                                    # to MOT format
                                    bbox_left = output[0]
                                    bbox_top = output[1]
                                    bbox_w = output[2] - output[0]
                                    bbox_h = output[3] - output[1]
                                    # Write MOT compliant results to file
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(('%g ' * 10 + '\n') % (count + 1, id, bbox_left,  # MOT format
                                                                       bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                                # if save_vid or save_crop or show_vid:  # Add bbox to image
                                if 1:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    id = int(id)  # integer id
                                    label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                        (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                    annotator.box_label(bboxes, label, color=colors(c, True))
                                        # --------------目标出现判断结束------------------------------------
                    prev_frames[i] = curr_frames[i]
                im0 = annotator.result()
                # 控制视频发送频率
                if self.rate_check:
                    time.sleep(1/self.rate)
                self.send_img.emit(im0)
                self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                self.send_statistic.emit(statistic_dic)
                # 如果自动录制
                if self.save_fold:
                    os.makedirs(self.save_fold, exist_ok=True)  # 路径不存在，自动保存
                    # 如果输入是图片
                    if self.vid_cap is None:
                        save_path = os.path.join(self.save_fold,
                                                 time.strftime('%Y_%m_%d_%H_%M_%S',
                                                               time.localtime()) + '.jpg')
                        cv2.imwrite(save_path, im0)
                    else:
                        if count == 1:  # 第一帧时初始化录制
                            # 以视频原始帧率进行录制
                            ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                            if ori_fps == 0:
                                ori_fps = 25
                            # width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            # height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            width, height = im0.shape[1], im0.shape[0]
                            save_path = os.path.join(self.save_fold, time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.mp4')
                            self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                       (width, height))
                        self.out.write(im0)
                if percent == self.percent_length:
                    print(count)
                    self.send_percent.emit(0)
                    self.send_msg.emit('检测结束')
                    if hasattr(self, 'out'):
                        self.out.release()
                    # 正常跳出循环
                    break


class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False
        # win10的CustomizeWindowHint模式，边框上面有一段空白。
        # 不想看到顶部空白可以用FramelessWindowHint模式，但是需要重写鼠标事件才能通过鼠标拉伸窗口，比较麻烦
        # 不嫌麻烦可以试试, 写了一半不想写了，累死人
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint )
        # self.setWindowFlags(Qt.FramelessWindowHint)
        # 自定义标题栏按钮
        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        self.closeButton.clicked.connect(self.close)

        # 定时清空自定义状态栏上的文字
        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # 自动搜索模型
        self.comboBox.clear()
        self.pt_list = os.listdir('./weights')
        # self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./weights/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5线程
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./weights/%s" % self.model_type           # 权重
        self.det_thread.source = '/home/cwh/桌面/SH-human-main/testvideo/fall_01.avi' # 默认打开本机摄像头，无需保存到配置文件
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.chose_cam)
        self.rtspButton.clicked.connect(self.chose_rtsp)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        # self.comboBox.currentTextChanged.connect(lambda x: self.statistic_msg('模型切换为%s' % x))
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.saveCheckBox.clicked.connect(self.is_save)
        self.load_setting()

    def search_pt(self):
        pt_list = os.listdir('./weights')
        # pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./weights/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def is_save(self):
        if self.saveCheckBox.isChecked():
            # 选中时
            self.det_thread.save_fold = './result'
        else:
            self.det_thread.save_fold = None

    def checkrate(self):
        if self.checkBox.isChecked():
            # 选中时
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.67:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='提示', text='请稍等，正在加载rtsp视频流', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('加载rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    def chose_cam(self):
        try:
            self.stop()
            # MessageBox的作用：留出2秒，让上一次摄像头安全release
            MessageBox(
                self.closeButton, title='提示', text='请稍等，正在检测摄像头设备', time=2000, auto=True).exec_()
            # 自动检测本机有哪些摄像头
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('加载摄像头：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    # 导入配置文件
    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            savecheck = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "check": check,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                check = 0
                savecheck = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                check = config['check']
                savecheck = config['savecheck']
        self.confSpinBox.setValue(iou)
        self.iouSpinBox.setValue(conf)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check          # 是否启用延时
        self.saveCheckBox.setCheckState(savecheck)
        self.is_save()                              # 是否自动保存

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
            self.det_thread.conf_thres = x/100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x*100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x/100)
            self.det_thread.iou_thres = x/100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)   # 3秒后自动清除

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "检测结束":
            self.saveCheckBox.setEnabled(True)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./weights/%s" % self.model_type
        self.statistic_msg('模型切换为%s' % x)

    def open_file(self):
        # source = QFileDialog.getOpenFileName(self, '选取视频或图片', os.getcwd(), "Pic File(*.mp4 *.mkv *.avi *.flv "
        #                                                                    "*.jpg *.png)")
        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, '选取视频或图片', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('加载文件：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            # 切换文件后，上一次检测停止
            self.stop()

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    # 继续/暂停
    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = '摄像头设备' if source.isnumeric() else source
            self.statistic_msg('正在检测 >> 模型：{}，文件：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('暂停')

    # 退出检测循环
    def stop(self):
        self.det_thread.jump_out = True
        self.saveCheckBox.setEnabled(True)

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            # QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        # self.setCursor(QCursor(Qt.ArrowCursor))

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # 保持纵横比
            # 找出长边
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # 实时统计
    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        # 如果摄像头开着，先把摄像头关了再退出，否则极大可能可能导致检测线程未退出
        self.det_thread.jump_out = True
        # 退出时，保存设置
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config['savecheck'] = self.saveCheckBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='提示', text='请稍等，正在关闭程序。。。', time=2000, auto=True).exec_()
        sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())

