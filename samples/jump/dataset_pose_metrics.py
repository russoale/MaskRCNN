# -*- coding: utf-8 -*-
"""
Created on 25 Jan 2019, 15:53

@author: einfalmo
"""


# PCK reference joints
from samples.jump.bisp_joint_order import JumpJointOrder, SwimJointOrder
from samples.jump.pose_metrics import pck_normalized_distances_fast

_jump_reference_joints = [JumpJointOrder.l_shoulder, JumpJointOrder.r_hip]
_swim_reference_joints = [SwimJointOrder.l_shoulder, SwimJointOrder.r_hip]


def pck_normalized_distances_jump(predictions, annotations, fallback_ref_lengths=None):
    """
    For documentation, see 'pck_normalized_distances_fast'.
    """
    return pck_normalized_distances_fast(predictions, annotations,
                                         ref_length_indices=_jump_reference_joints,
                                         fallback_ref_lengths=fallback_ref_lengths)


def pck_normalized_distances_swim(predictions, annotations, fallback_ref_lengths=None):
    """
    For documentation, see 'pck_normalized_distances_fast'.
    """
    return pck_normalized_distances_fast(predictions, annotations,
                                         ref_length_indices=_swim_reference_joints,
                                         fallback_ref_lengths=fallback_ref_lengths)
