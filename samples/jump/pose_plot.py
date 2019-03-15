#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:05:23 2016

@author: einfalmo

Collection of plot and imshow tools for human poses.
"""

import cv2
import numpy as np
import colorsys
from data_handling import bisp_joint_order


def _hsv_colors(num_colors):
    """
    Taken from: http://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors 
    (Uri Cohen)
    """
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = (60 / 100.0)
        saturation = (95 / 100.0)
        color = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(tuple([int(c * 255) for c in color]))
    return colors


class SwimBodypartColors:
    __num_body_parts_colors = 6
    __raw_hsv_colors = _hsv_colors(__num_body_parts_colors)

    head = __raw_hsv_colors[2]
    torso = __raw_hsv_colors[0]
    r_upper_arm = __raw_hsv_colors[1]
    r_lower_arm = __raw_hsv_colors[5]
    l_upper_arm = __raw_hsv_colors[1]
    l_lower_arm = __raw_hsv_colors[5]
    r_upper_leg = __raw_hsv_colors[3]
    r_lower_leg = __raw_hsv_colors[4]
    l_upper_leg = __raw_hsv_colors[3]
    l_lower_leg = __raw_hsv_colors[4]

    colors = [head, torso,
              r_upper_arm, r_lower_arm, l_upper_arm, l_lower_arm,
              r_upper_leg, r_lower_leg, l_upper_leg, l_lower_leg]

    stroked = [False, False,
               False, False, True, True,
               False, False, True, True]


    def __init__(self):
        pass


class JumpBodypartColors:
    __num_body_parts_colors = 10
    __raw_hsv_colors = _hsv_colors(__num_body_parts_colors)

    head = __raw_hsv_colors[0]
    torso = __raw_hsv_colors[4]
    r_upper_arm = __raw_hsv_colors[6]
    r_lower_arm = __raw_hsv_colors[9]
    r_hand = __raw_hsv_colors[2]
    l_upper_arm = __raw_hsv_colors[6]
    l_lower_arm = __raw_hsv_colors[9]
    l_hand = __raw_hsv_colors[2]
    r_upper_leg = __raw_hsv_colors[1]
    r_lower_leg = __raw_hsv_colors[8]
    r_ankle = __raw_hsv_colors[5]
    r_foot = __raw_hsv_colors[3]
    l_upper_leg = __raw_hsv_colors[1]
    l_lower_leg = __raw_hsv_colors[8]
    l_ankle = __raw_hsv_colors[5]
    l_foot = __raw_hsv_colors[3]
    passpoint = __raw_hsv_colors[7]

    colors = [head, torso,
              r_upper_arm, r_lower_arm, r_hand,
              l_upper_arm, l_lower_arm, l_hand,
              r_upper_leg, r_lower_leg, r_ankle, r_foot,
              l_upper_leg, l_lower_leg, l_ankle, l_foot,
              passpoint]

    stroked = [False, False,
               False, False, False,
               True, True, True,
               False, False, False, False,
               True, True, True, True,
               False]


    def __init__(self):
        pass


def _draw_line(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    """
    Util to draw stroked or dotted lines
    
    Taken from: http://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
    (User: Zaw Lin)
    """
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1, lineType=cv2.LINE_AA)
    else:
        if len(pts) > 0:
            s = pts[0]
            e = pts[0]
            i = 0
            for p in pts:
                s = e
                e = p
                if i % 2 == 1:
                    cv2.line(img, s, e, color, thickness, lineType=cv2.LINE_AA)
                i += 1


def draw_pose(image, joint_annotations, joint_order, bodypart_colors, draw_into_original=False, thickness=3,
              draw_invalid_joints=True):
    """
    Draws the humaon pose defined by the joint locations onto the image.
    
    Parameters:
        - image: numpy.ndarray
            The original image
        - joint_annotations: numpy.ndarray
            Joint locations (num_joints x 2)
        - draw_into_original: Whether to draw into the original image or create a copy.
    """

    if draw_into_original:
        out_image = image
    else:
        out_image = np.copy(image)

    body_parts = joint_order.joints_to_bodyparts(joint_annotations)
    for i, part in enumerate(body_parts):
        part_begin, part_end = (int(part[0, 0]), int(part[0, 1])), (int(part[1, 0]), int(part[1, 1]))
        if i != 1 and part_begin[0] >= 0 and part_begin[1] >= 0:
            cv2.circle(out_image, part_begin, 5, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        if i != 1 and part_end[0] >= 0 and part_end[1] >= 0:
            cv2.circle(out_image, part_end, 5, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        if draw_invalid_joints or all([coord >= 0 for coord in list(part_begin) + list(part_end)]):
            color = bodypart_colors.colors[i]
            stroked = bodypart_colors.stroked[i]
            if stroked:
                _draw_line(out_image, part_end, part_begin, color, thickness=thickness, style='stroked', gap=6)
            else:
                cv2.line(out_image, tuple(part_begin), tuple(part_end), color, thickness=thickness,
                         lineType=cv2.LINE_AA)
    return out_image


def draw_swim_pose(image, joint_annotations, draw_into_original=False, thickness=3, draw_invalid_joints=False):
    draw_pose(image, joint_annotations, bisp_joint_order.SwimJointOrder, SwimBodypartColors, draw_into_original,
              thickness, draw_invalid_joints)


def draw_jump_pose(image, joint_annotations, draw_into_original=False, thickness=3, draw_invalid_joints=False):
    draw_pose(image, joint_annotations, bisp_joint_order.JumpJointOrder, JumpBodypartColors, draw_into_original,
              thickness, draw_invalid_joints)
