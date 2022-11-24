'''
Descripttion:  判定动作
version:
Author: LiQiang
Date: 2021-06-06 09:14:31
LastEditTime: 2021-09-24 19:57:37
'''

# 图像坐标归一化

import myutils.positionCal as positionCal


def normalize_coordinates(row_i, col_j, img):
    num_rows, num_cols = img.shape[:2]
    x = col_j / (num_cols - 1.)
    y = row_i / (num_rows - 1.)
    return x, y

def isContact(pose_results):
    try:
        person1 = pose_results[0]['keypoints'].tolist()
        person2 = pose_results[1]['keypoints'].tolist()
        persons = {'1': person1, '2':person2}
        return positionCal.isContact(persons)
    except:
        return False


def isBeishuai(pose_results):
    try:
        person1 = pose_results[0]['keypoints'].tolist()
        person2 = pose_results[1]['keypoints'].tolist()
        persons = {'1': person1, '2':person2}

        return positionCal.ishand2neck(persons) and positionCal.isHSParallel(persons)
    # return positionCal.isHSParallel(persons)
    except:
        return False

def isTuisang(pose_results):
    try:
        person1 = pose_results[0]['keypoints'].tolist()
        person2 = pose_results[1]['keypoints'].tolist()
        persons = {'1': person1, '2':person2}

        return positionCal.ishand2body(persons) and positionCal.isInbody(persons)
    # return positionCal.isHSParallel(persons)
    except:
        return False

#
# def isfight(pose_results):
#     #
#     person1 = pose_results[0]['keypoints'].tolist()
#     # person2 = pose_results[1]['keypoints'].tolist()
#     # 人1
#     x1_6, y1_6 = person1[6][0], person1[6][1]
#     x1_8, y1_8 = person1[8][0], person1[8][1]
#     x1_7, y1_7 = person1[7][0], person1[7][1]
#     x1_9, y1_9 = person1[9][0], person1[9][1]
#     x1_10, y1_10 = person1[10][0], person1[10][1]
#     x1_11, y1_11 = person1[11][0], person1[11][1]
#     # 人2
#     # x2_6, y2_6 = person2[6][0], person2[6][1]
#     # x2_8, y2_8 = person2[8][0], person2[8][1]
#     # x2_7, y2_7 = person2[7][0], person2[7][1]
#     # x2_9, y2_9 = person2[9][0], person2[9][1]
#     # x2_10, y2_10 = person2[10][0], person2[10][1]
#     # x2_11, y2_11 = person2[11][0], person2[11][1]
#     # print(len(keypoints))
#
#     k_left = abs((y1_11 - y1_7) / ((x1_11 - x1_7) + 2e-18))
#     k_right = abs((y1_10 - y1_6) / ((x1_10 - x1_6) + 2e-18))
#
#     if k_right < 0.5:
#         return True
#     else:
#         return False
#

'''
Descripttion:  判定动作
version: 
Author: LiQiang
Date: 2021-06-06 09:14:31
LastEditTime: 2021-09-24 19:57:37
'''




# 举手动作识别
def is_raise_hand(poseresults):
    """
    poseresults: 关节点信息
    [{'bbox': array([        152,         145,         442,         679,      0.8188]), 
    'track_id': 1.0, 'keypoints': array([[     315.25,      190.37,     0.94378],
       [     336.11,      179.94,     0.93666],
       [     299.61,      174.72,     0.93834],
       [     356.97,      195.58,     0.94361],
       [     273.53,      185.15,      0.9406],
       [     383.04,      284.24,     0.86726],
       [     221.38,      284.24,     0.80752],
       [     409.12,      393.75,     0.68802],
       [     195.31,      409.39,     0.66317],
       [     393.47,      435.47,     0.60395],
       [      330.9,       445.9,     0.45193],
       [     356.97,      471.97,     0.52449],
       [     252.67,      471.97,     0.47213],
       [      471.7,      539.76,     0.48057],
       [     315.25,      466.76,     0.12565],
       [     461.27,      690.99,     0.37615],
       [     242.24,      612.77,    0.058486]], dtype=float32)},
    """
    res = []
    
    try:
    # 左手
        for person in poseresults:
            keypoints = person['keypoints'].tolist()
            track_id = person['track_id']
            bbox = person['bbox']
            x8,y8=keypoints[7][0],keypoints[7][1]
            x10,y10=keypoints[9][0],keypoints[9][1]

            # 右手
            x9,y9=keypoints[8][0],keypoints[8][1]
            x11,y11=keypoints[10][0],keypoints[10][1]

            # 判断斜率 【>4.0】
            k_right=abs((y11-y9)/((x11-x9)+2e-18))
            k_left=abs((y10-y8)/((x10-x8)+2e-18))
            if y11<y9 or y10<y8:  # 判断关节11 点要在关节9之上
                if  k_right>4.0 or k_left>4.0:
                    res.append({"track_id":track_id,"bbox":bbox})
        return res           
    except:
        return res
    

def is_stand_up(poseresults):
    """
    判断是否站起 
    关节点 12 14   左侧
    关节点 13 15   右侧
    """
    res = []
    try:
        for person in poseresults:
            keypoints = person['keypoints'].tolist()
            track_id = person['track_id']
            bbox = person['bbox']
            x12,y12=keypoints[11][0],keypoints[11][1]
            x14,y14=keypoints[13][0],keypoints[13][1]

            x13,y13=keypoints[12][0],keypoints[12][1]
            x15,y15=keypoints[14][0],keypoints[14][1]

            # 判断斜率 【>6.0】
            k_right=abs((y14-y12)/((x14-x12)+2e-18))
            k_left=abs((y15-y13)/((x15-x13)+2e-18))
            if y13<y15 or y12<y14:
                if  k_right>6.0 and k_left>6.0:
                    res.append({"track_id":track_id,"bbox":bbox})
        return res
    except:
        return res

    

