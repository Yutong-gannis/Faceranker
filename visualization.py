# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 19:41:59 2022

@author: lenovo
"""

import  mediapipe as mp

def face_network(img, mp_drawing, mp_face_mesh, results, color):
    draw_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1,color=color)

    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=img,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=draw_spec,
            connection_drawing_spec=draw_spec)
    return img

def face_outline(img, mp_drawing, mp_face_mesh, results, color):
    draw_spec=mp_drawing.DrawingSpec(thickness=2,circle_radius=1,color=color)
    
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=img,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=draw_spec)
    return img

def pupil(img, mp_drawing, mp_face_mesh, results, color):
    draw_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1,color=color)
    
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=img,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=draw_spec)
    return img
