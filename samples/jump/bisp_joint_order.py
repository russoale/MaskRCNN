#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 08:32:00 2016

@author: einfalmo
"""

import numpy as np


class SwimJointOrder:
    
    head = 0
    neck = 1
    r_shoulder = 2
    r_elbow = 3
    r_wrist = 4
    l_shoulder = 5
    l_elbow = 6
    l_wrist = 7
    r_hip = 8
    r_knee = 9
    r_ankle = 10
    l_hip = 11
    l_knee = 12
    l_ankle = 13
    
    num_joints = 14
    num_bodyparts = 10

    @classmethod
    def indices(cls):
        return [cls.head, cls.neck, 
                cls.r_shoulder, cls.r_elbow, cls.r_wrist,
                cls.l_shoulder, cls.l_elbow, cls.l_wrist,
                cls.r_hip, cls.r_knee, cls.r_ankle,
                cls.l_hip, cls.l_knee, cls.l_ankle]
                
    @classmethod
    def bodypart_indices(cls):
        return[[cls.head, cls.neck], 
               [cls.r_shoulder, cls.r_elbow], [cls.r_elbow, cls.r_wrist],
               [cls.l_shoulder, cls.l_elbow], [cls.l_elbow, cls.l_wrist],
               [cls.r_hip, cls.r_knee], [cls.r_knee, cls.r_ankle],
               [cls.l_hip, cls.l_knee], [cls.l_knee, cls.l_ankle]]
               
    @classmethod
    def names(cls):
        return ["head", "neck", 
               "rsho", "relb", "rwri",
               "lsho", "lelb", "lwri",
               "rhip", "rkne", "rank",
               "lhip", "lkne", "lank"]
        
    @classmethod
    def pretty_names(cls):
        return ["Head", "Neck",
                "R. shoulder", "R. elbow", "R. wrist",
                "L. shoulder", "L. elbow", "L. wrist",
                "R. hip", "R. knee", "R. ankle",
                "L. hip", "L. knee", "L. ankle"]

    @classmethod
    def bodypart_names(cls):
        return ["head",
                "ruarm", "rlarm", "luarm", "llarm",
                "ruleg", "rlleg", "luleg", "llleg"]
    
    def __init__(self):
        pass
    
    @classmethod
    def joints_to_bodyparts(cls, joint_annotation):
        has_visibility_flag = (joint_annotation.shape[1] == 3)
        if has_visibility_flag:
            joint_dim = 3
        else:
            joint_dim = 2
        bodyparts = np.empty((cls.num_bodyparts, 2, joint_dim), dtype=np.float32)
        for i, indices in enumerate(cls.bodypart_indices()):
            if i > 0:
                j = i+1
            else:
                j = i
            bodyparts[j] = joint_annotation[indices]
        # Special joint: Torso
        torso_upper = (joint_annotation[cls.l_shoulder] + joint_annotation[cls.r_shoulder]) / 2.0
        torso_lower = (joint_annotation[cls.l_hip] + joint_annotation[cls.r_hip]) / 2.0
        if has_visibility_flag:
            if joint_annotation[cls.l_shoulder][2] == -1 or joint_annotation[cls.r_shoulder][2] == -1:
                torso_upper[:] = [-1., -1., 0]
            else:
                torso_upper[2] = 1
            
            if joint_annotation[cls.l_hip][2] == -1 or joint_annotation[cls.r_hip][2] == -1:
                torso_lower[:] = [-1., -1., 0]
            else:
                torso_lower[2] = 1
        bodyparts[1, 0] = torso_upper
        bodyparts[1, 1] = torso_lower
    
        return bodyparts
    
    
    
class JumpJointOrder:

    head = 0
    neck = 1
    r_shoulder = 2
    r_elbow = 3
    r_wrist = 4
    r_hand = 5
    l_shoulder = 6
    l_elbow = 7
    l_wrist = 8
    l_hand = 9
    r_hip = 10
    r_knee = 11
    r_ankle = 12
    r_heel = 13
    r_toetip = 14
    l_hip = 15
    l_knee = 16
    l_ankle = 17
    l_heel = 18
    l_toetip = 19
 
    # r_toetip = 0
    # r_heel = 1
    # r_ankle = 2
    # r_knee = 3
    # r_hip = 4
    # l_hip = 5
    # l_knee = 6
    # l_ankle = 7
    # l_heel = 8
    # l_toetip = 9
    # r_hand = 10
    # r_wrist = 11
    # r_elbow = 12
    # r_shoulder = 13
    # l_shoulder = 14
    # l_elbow = 15
    # l_wrist = 16
    # l_hand = 17
    # neck = 18
    # head = 19
    #pass1 = 20
    #pass2 = 21
    
    num_joints = 20
    num_bodyparts = 16

    @classmethod
    def indices(cls):
        return [cls.head, cls.neck,
                cls.r_shoulder, cls.r_elbow, cls.r_wrist, cls.r_hand,
                cls.l_shoulder, cls.l_elbow, cls.l_wrist, cls.l_hand,
                cls.r_hip, cls.r_knee, cls.r_ankle, cls.r_heel, cls.r_toetip,
                cls.l_hip, cls.l_knee, cls.l_ankle, cls.l_heel, cls.l_toetip]

    @classmethod
    def bodypart_indices(cls):
        return [[cls.head, cls.neck], 
                [cls.r_shoulder, cls.r_elbow], [cls.r_elbow, cls.r_wrist], [cls.r_wrist, cls.r_hand],
                [cls.l_shoulder, cls.l_elbow], [cls.l_elbow, cls.l_wrist], [cls.l_wrist, cls.l_hand],
                [cls.r_hip, cls.r_knee], [cls.r_knee, cls.r_ankle], [cls.r_ankle, cls.r_heel], [cls.r_heel, cls.r_toetip],
                [cls.l_hip, cls.l_knee], [cls.l_knee, cls.l_ankle], [cls.l_ankle, cls.l_heel], [cls.l_heel, cls.l_toetip]]
               
    @classmethod
    def names(cls):
        return ["head", "neck",
               "rsho", "relb", "rwri", "rhan",
               "lsho", "lelb", "lwri", "lhan",
               "rhip", "rkne", "rank", "rhee", "rtoe",
               "lhip", "lkne", "lank", "lhee", "ltoe"]
        
    @classmethod
    def pretty_names(cls):
        return ["Head", "Neck",
                "R. shoulder", "R. elbow", "R. wrist", "R. hand",
                "L. shoulder", "L. elbow", "L. wrist", "L. hand",
                "R. hip", "R. knee", "R. ankle", "R. heel", "R. toetip",
                "L. hip", "L. knee", "L. ankle", "L. heel", "L. toetip"]
    
    def __init__(self):
        pass
    
    @classmethod
    def joints_to_bodyparts(cls, joint_annotation):
        has_visibility_flag = (joint_annotation.shape[1] == 3)
        if has_visibility_flag:
            joint_dim = 3
        else:
            joint_dim = 2
        bodyparts = np.empty((cls.num_bodyparts, 2, joint_dim), dtype=np.float32)
        for i, indices in enumerate(cls.bodypart_indices()):
            if i > 0:
                j = i + 1
            else:
                j = i
            bodyparts[j] = joint_annotation[indices]
        # Special joint: Torso
        torso_upper = (joint_annotation[cls.l_shoulder] + joint_annotation[cls.r_shoulder]) / 2.0
        torso_lower = (joint_annotation[cls.l_hip] + joint_annotation[cls.r_hip]) / 2.0
        if has_visibility_flag:
            if joint_annotation[cls.l_shoulder][2] == -1 or joint_annotation[cls.r_shoulder][2] == -1:
                torso_upper[:] = [-1., -1., 0]
            else:
                torso_upper[2] = 1

            if joint_annotation[cls.l_hip][2] == -1 or joint_annotation[cls.r_hip][2] == -1:
                torso_lower[:] = [-1., -1., 0]
            else:
                torso_lower[2] = 1
        bodyparts[1, 0] = torso_upper
        bodyparts[1, 1] = torso_lower

        return bodyparts
