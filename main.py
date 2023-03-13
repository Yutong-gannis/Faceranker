# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 18:09:09 2022

@author: lenovo
"""

import cv2
import  mediapipe as mp
import time
import torch

from visualization import face_network, face_outline, pupil
from mark import WuYan_index, SanTing_index, DaVinci, deep_learning
from forward import load_model
import Nets

# 导入三维人脸关键点检测模型
mp_face_mesh=mp.solutions.face_mesh

model=mp_face_mesh.FaceMesh(
    static_image_mode=False,  # TRUE:静态图片/False:摄像头实时读取
    refine_landmarks=True,  # 使用Attention Mesh模型
    max_num_faces=10,
    min_detection_confidence=0.75,  # 置信度阈值，越接近1越准
    min_tracking_confidence=0.5,  # 追踪阈值
)


# 导入可视化函数和可视化样式
mp_drawing=mp.solutions.drawing_utils
landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=[233, 155, 6])
# 轮廓可视化
connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=[233, 155, 6])


# 处理帧函数
def process_frame(img, net):
    start_time = time.time()
    scaler = 1
    radius = 6
    lw = 2
    img = cv2.resize(img, (1280, 960)) # BGR通道
    h, w = img.shape[0], img.shape[1]
    blue = (225, 238, 160)
    white = (255, 255, 255)
    gray = (132,133,135)
    green = (0, 255, 127)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转RGB通道
    results = model.process(img_RGB) # 所有结果
    
    i = 0
    if results.multi_face_landmarks:
        img = face_network(img, mp_drawing, mp_face_mesh, results, blue) # 显示脸部网络
        img = face_outline(img, mp_drawing, mp_face_mesh, results, white) # 显示脸部轮廓
        img = pupil(img, mp_drawing, mp_face_mesh, results, white) # 显示瞳孔
        for face_landmarks in results.multi_face_landmarks:
            # 五眼指标
            img, Five_Eye_Metrics, Five_Eye_Diff, FL_X, FL_Y, FT_X, FT_Y, FB_X, FB_Y, FR_X, FR_Y, ELL_X, ELL_Y, ELR_X, ELR_Y, ERL_X, ERL_Y, ERR_X, ERR_Y, Left_Right = WuYan_index(img, face_landmarks, h, w)
            # 三庭
            img, Three_Section_Mrtric_A, Three_Section_Mrtric_B, Three_Section_Mrtric_C = SanTing_index(img, face_landmarks, FT_Y, FB_Y, FL_X, FR_X, h, w)
            # 达芬奇
            img, Da_Vinci = DaVinci(img, face_landmarks, ELR_X, ELR_Y, ERL_X, ERL_Y, ELL_X, ELL_Y, ERR_X, ERR_Y, Left_Right, h, w)
            # ResNet预测
            score = deep_learning(img_RGB, net, FL_X, FR_X, FT_Y, FB_Y, h, w)
            # 可视化
            img = cv2.putText(img, str(i), (FL_X, FT_Y-90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             green, 2)
            img = cv2.putText(img, 'ResNet-18 Score{:' '>7.1f}'.format(score), (FL_X, FT_Y-70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             green, 2)
            img = cv2.putText(img, 'Five Eye Metrics{:' '>7.2f}'.format(Five_Eye_Metrics), (FL_X, FT_Y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             green, 2)
            '''
            img = cv2.putText(img, 'A{:' '>7.2f}'.format(Five_Eye_Diff[0]), (FL_X, FT_Y-105), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             green, 2)
            img = cv2.putText(img, 'B{:' '>7.2f}'.format(Five_Eye_Diff[2]), (FL_X, FT_Y-90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             green, 2)
            img = cv2.putText(img, 'C{:' '>7.2f}'.format(Five_Eye_Diff[4]), (FL_X, FT_Y-75), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             green, 2)
            '''
            img = cv2.putText(img, 'Three Scetion{:' '>7.2f}'.format(Three_Section_Mrtric_A), (FL_X, FT_Y-30), cv2.FONT_HERSHEY_SIMPLEX,
                             0.5,
                             green, 2)
            '''
            img = cv2.putText(img, '1/3{:' '>7.2f}'.format(Three_Section_Mrtric_B), (FL_X, FT_Y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             green, 2)
            img = cv2.putText(img, '1/2{:' '>7.2f}'.format(Three_Section_Mrtric_C), (FL_X, FT_Y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             green, 2)
            '''
            img = cv2.putText(img, 'Da Vinci{:' '>7.2f}'.format(Da_Vinci), (FL_X, FT_Y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             green, 2)
            i = i + 1

    else:
        img = cv2.putText(img, 'NO FACE DELECTED', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                         green, 3)

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像的帧数FPS
    FPS = 1 / (end_time - start_time)
    scaler = 1
    img = cv2.putText(img, 'FPS{:' '>7.2f}'.format(FPS), (25 * scaler, 700 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                         1.25 * scaler, green, 3, 10)
    return img


#net = Nets.AlexNet().cuda()
net = Nets.ResNet(block = Nets.BasicBlock, layers = [2, 2, 2, 2], num_classes = 1).cuda()
# load_model(torch.load('./models/alexnet.pth'), net)
load_model(torch.load('./models/resnet18.pth', encoding='ISO-8859-1'), net)
net.eval()
print('Model loaded')
# 调用摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.open(0)
# 无限循环，直到break被触发
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print('ERROR')
        break
    frame = process_frame(frame, net)
    cv2.imshow('my_window', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
