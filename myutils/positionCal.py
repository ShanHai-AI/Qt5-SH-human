import numpy as np

# 获得当前人体的全部关节点中最左侧x坐标
def getLeft(keypoints):

    list = []
    for i in range(17):
        list.append(keypoints[i][0])
    return np.array(list).min()

# 获得当前人体的全部关节点中最右侧x坐标
def getRight(keypoints):

    list = []
    for i in range(17):
        list.append(keypoints[i][0])
    return np.array(list).max()

# 获得当前人体的头部高度：（颈部-眼部的y坐标最低，最高）
def getHead(keypoints):

    list = []
    for i in range(7):
        list.append(keypoints[i][1])
    # print("head:{}-{}".format(np.array(list).min(), np.array(list).max()))
    return (np.array(list).min(), np.array(list).max())

# 获得当前人体的上半身高度：（颈部-腰部的y坐标最低，最高）
def getUpperbodyHeight(keypoints):

    list = []
    for i in range(5, 13):
        list.append(keypoints[i][1])
    # print("head:{}-{}".format(np.array(list).min(), np.array(list).max()))
    return (np.array(list).min(), np.array(list).max())


# 获得当前人体的上半身宽度：（手部-肩部的x坐标最左，最右）
def getUpperbodyWidth(keypoints):
    list = []
    for i in range(5, 13):
        list.append(keypoints[i][0])
    return (np.array(list).min(), np.array(list).max())


# 判断两人是否有身体接触
def isContact(persons):

    p1_left = getLeft(persons['1'])
    p2_left = getLeft(persons['2'])
    p1_right = getRight(persons['1'])
    p2_right = getRight(persons['2'])
    if (p2_left < p1_right and p2_right > p1_left) or (p1_left < p2_right and p1_right > p2_left):
        return True
    return False

# 判断 两人接触时 是否有人的手接触到另一人头部
def ishand2neck(persons):

    p1_head = getHead(persons['1'])
    p2_head = getHead(persons['2'])

    if p1_head[0] <= persons['2'][10][1] <=p1_head[1]:
        return True
    elif p1_head[0] <= persons['2'][9][1] <=p1_head[1]:
        return True
    elif p2_head[0] <= persons['1'][10][1] <=p2_head[1]:
        return True
    elif p2_head[0] <= persons['1'][9][1] <=p2_head[1]:
        return True
    else:return False

# 判断两条直线的相似度：小于参数nearness的视为近似平行
def isNearlyParallel(l1, l2, nearness = 0.05):

    slop1 = (l1[1][1]-l1[0][1])/(l1[1][0] - l1[0][0]+ 1e-5 )
    slop2 = (l2[1][1]-l2[0][1])/(l2[1][0] - l2[0][0]+ 1e-5 )
    if (abs(slop1) - abs(slop2))<= nearness:
        return True
    return False

# 判断P1手肩连线与P2是否平行
def isHSParallel(persons):

    p1_left_hs =[persons['1'][10][0:2], persons['1'][6][0:2]]
    p1_right_hs =[persons['1'][9][0:2], persons['1'][5][0:2]]
    p2_left_hs =[persons['2'][10][0:2], persons['2'][6][0:2]]
    p2_right_hs =[persons['2'][9][0:2], persons['2'][5][0:2]]
    p1_shoulder =[persons['1'][6][0:2], persons['1'][5][0:2]]
    p2_shoulder =[persons['2'][6][0:2], persons['2'][5][0:2]]

    return isNearlyParallel(p1_left_hs, p2_shoulder) or \
           isNearlyParallel(p1_right_hs, p2_shoulder) or \
           isNearlyParallel(p2_left_hs, p1_shoulder) or \
           isNearlyParallel(p2_right_hs, p1_shoulder)

# 判断 两人接触时  是否P1手部接触P2身体
def ishand2body(persons):
    p1_up_h = getUpperbodyHeight(persons['1'])
    p2_up_h = getUpperbodyHeight(persons['2'])

    if p1_up_h[0] <= persons['2'][10][1] <=p1_up_h[1]:
        return True
    elif p1_up_h[0] <= persons['2'][9][1] <=p1_up_h[1]:
        return True
    elif p2_up_h[0] <= persons['1'][10][1] <=p2_up_h[1]:
        return True
    elif p2_up_h[0] <= persons['1'][9][1] <=p2_up_h[1]:
        return True
    else:return False

# 判断 两人接触时 是否P1手部深入P2身体区间(在手和肩对应的x坐标内)
def isInbody(persons, theta = 0.2):
    p1_up_w = getUpperbodyWidth(persons['1'])
    p2_up_w = getUpperbodyWidth(persons['2'])

    if (p1_up_w[0]+(p1_up_w[1]-p1_up_w[0])*theta) <= persons['2'][10][0] <=(p1_up_w[1]- (p1_up_w[1]-p1_up_w[0])*theta):
        return True
    elif (p1_up_w[0]+(p1_up_w[1]-p1_up_w[0])*theta) <= persons['2'][9][0] <=(p1_up_w[1]- (p1_up_w[1]-p1_up_w[0])*theta):
        return True
    elif (p2_up_w[0]+ (p2_up_w[1]-p2_up_w[0])*theta) <= persons['1'][10][0] <=(p2_up_w[1]- (p2_up_w[1]-p2_up_w[0])*theta):
        return True
    elif (p2_up_w[0]+ (p2_up_w[1]-p2_up_w[0])*theta)  <= persons['1'][9][0] <=(p2_up_w[1]- (p2_up_w[1]-p2_up_w[0])*theta):
        return True
    else:return False

    pass

if __name__ == '__main__':
    keypoints =     \
        [[     979.34 ,    269.72 ,    0.94007],
        [     1000.6 ,     248.47  ,   0.95547],
        [     952.78   ,   248.47   ,    0.922],
        [     1021.8 ,     248.47 ,    0.88961],
        [     904.97   ,   253.78  ,   0.91527],
        [       1059  ,    328.16,     0.79805],
        [     862.47 ,     338.78  ,   0.72687],
        [     1138.7   ,   450.34,     0.63871],
        [     819.97  ,    460.97   ,  0.81358],
        [     1112.2   ,   514.09 ,     0.8293],
        [     915.59  ,    519.41,     0.90664],
        [     1032.5 ,     551.28  ,   0.50632],
        [     904.97  ,    556.59 ,      0.396],
        [     1165.3    ,  530.03   ,  0.16438],
        [     835.91    ,  530.03 ,    0.28814],
        [     1032.5  ,    588.47 ,   0.092156],
        [     1043.1  ,    630.97 ,    0.21141]]


    l1 = [keypoints[10][0:2], keypoints[6][0:2]]
    print(l1)