'''
Descripttion: 
version: 
Author: LiQiang
Date: 2022-09-29 14:36:17
LastEditTime: 2022-09-29 16:00:53
'''
import os
from argparse import ArgumentParser

from xtcocotools.coco import COCO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from alfred.dl.torch.common import device
import cv2
import time
import torch


def process_mmdet_results(mmdet_results, cat_id=0):
    """AI is creating summary for process_mmdet_results

    Args:
        mmdet_results ([type]):目标检测结果 [ [x1,y1,x2,y2,id,class,p],[x1,y1,x2,y2,id,class,p],[x1,y1,x2,y2,id,class,p]]
        cat_id (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    person_results = []
    for bbox in mmdet_results:
        person = {}
        bbox=list(bbox)
        person['bbox'] = bbox[0:4]
        person['bbox'].append(bbox[-1])
        # print(person['bbox'])
        person['track_id']=bbox[4]
        person_results.append(person)
        # print(person)
    return person_results

def init_model():
    '''
    parser = ArgumentParser()
    parser.add_argument('--pose_config', default='mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv2_coco_256x192.py',help='Config file for detection')
    parser.add_argument('--pose_checkpoint',default='mmpose/weights/shufflenetv2_coco_256x192-0aba71c7_20200921.pth',help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
        pose_model = init_pose_model(
        args.pose_config, 
        args.pose_checkpoint, 
        device=args.device.lower()
        )
    '''
    # build the pose model from a config file and a checkpoint file
    pose_config='mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv2_coco_256x192.py'
    pose_checkpoint='mmpose/weights/shufflenetv2_coco_256x192-0aba71c7_20200921.pth'
    device='cuda:0'
    pose_model = init_pose_model(
        pose_config, 
        pose_checkpoint, 
        device=device.lower()
        )
    return pose_model
pose_model=init_model()

def skeleton_detect(mot_res,frame,kpt_thr=0.3):
    """AI is creating summary for skeleton_detect

    Args:
        mot_res ([List]): 目标检测结果  x1,y1,x2,y2,id,class
        frame:当前视频帧 cv2 numpy
        kpt_thr:Keypoint score threshold
    """
    dataset = pose_model.cfg.data['test']['type']
    return_heatmap = False
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    # keep the person class bounding boxes.
    person_results = process_mmdet_results(mot_res)
    # test a single image, with a list of bboxes.
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        frame,
        person_results,
        bbox_thr=0.3,
        format='xyxy',
        dataset=dataset,
        return_heatmap=return_heatmap,
        outputs=output_layer_names)

    # 1123 返回字典格式
    pose_results_dict = dict()
    for person in pose_results:
        pose_results_dict[person["track_id"]] = {
            "bbox":person["bbox"],
            "keypoints":person["keypoints"]
        }

    # show the results
    vis_img = vis_pose_result(
        pose_model,
        frame,
        pose_results,
        dataset=dataset,
        kpt_score_thr=kpt_thr,
        show=False)
    return pose_results, vis_img, pose_results_dict

