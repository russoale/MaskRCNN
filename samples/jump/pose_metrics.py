#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 08:57:39 2016

@author: einfalmo

Collection of helpers to calculate pose estimation metrics.
"""

import numpy as np
import math
import warnings
from data_handling.bisp_joint_order import JumpJointOrder

    
def pck_normalized_distances(predictions, annotations):
    """
    DEPRECATED. Use pck_normalized_distances_fast instead
    Calculates the normalized distances according to PCK metric between joint predicitons and annotations.
    
    Parameters:
        - predictions: numpy.ndarray
            2d array (num_predictions x 20(joints) x 2(x,y)).
        - annotations: numpy.ndarray
            2d array (num_predictions x 20(joints) x 2(x,y)).
    Returns:
        - numpy.ndarray
            2d array (num_predictons x num_distances(joints)).
    """
    warnings.warn("Use of outdated function 'pck_normalized_distances'. Use 'pck_normalized_distances_fast' instead. ")
    assert(predictions.shape == annotations.shape)
    num_predictions = predictions.shape[0]
    # Joint-related metrics
    joint_indices = JumpJointOrder.indices()
    distances = np.ndarray(shape=(num_predictions, len(joint_indices)), dtype=np.float32)
    for i in range(num_predictions):
        prediction = predictions[i, :, :]
        annotation = annotations[i, :, :]
        # Reference is distance from left shoulder to right hip
        # Use in quadratic form to enable fast sqrt in numpy at the end
        ref_length = float((annotation[JumpJointOrder.l_shoulder, 0] - annotation[JumpJointOrder.r_hip, 0]) ** 2 +
                           (annotation[JumpJointOrder.l_shoulder, 1] - annotation[JumpJointOrder.r_hip, 1]) ** 2)
        assert ref_length != 0
        for j, joint in enumerate(joint_indices):
            error = (annotation[joint, 0] - prediction[joint, 0]) ** 2 + \
                                (annotation[joint, 1] - prediction[joint, 1]) ** 2
            normalized_error = float(error) / ref_length
            distances[i, j] = normalized_error

    distances = np.sqrt(distances)

    return distances


def pck_normalized_distances_fast(predictions, annotations, ref_length_indices,
                                  fallback_ref_lengths=None):
    """
    Calculates the normalized distances according to PCK-like metrics between joint predictions and annotations.
    The PCK-like reference distance is specified by the given pair of joint indices.
    Optionally, fallback reference distances can be specified for each individual pose.
    This is necessary, when the reference distance using the joint-pair is ill defined for some poses.

    This function operates in the following way:
    If predictions and annotations are only 2D (x,y), the reference length will be blindly calculated for each pose,
    and every single joint prediction + annotations pair is converted into a normalized distances.
    For this to work, you have to ensure that only valid annotations are given to the function, i.e. all invalid cases
    have to be filtered out beforehand.

    If predictions and annotations are 3D (x,y,v), then the following will be applied:
    The v-flag is interpreted as "not available" if =0 and as "available" if >0.
    Joints that w.r.t the annotation are "not available" are assigned a normalized distance of 0.
    Since this can add a positive bias to any accumulated PCK result, you have to do a correct
    normalization of the PCK results yourself!
    If the annotation of a reference joint is "not available", either the fallback reference length is used (if given),
    or all distances for that pose are set to -1. In the second case it is again your job to handle these results
    appropriately.
    If the annotation of any other joint is "not available",
    then the distance of the respective joint is again set to -1.


    :param predictions: numpy.ndarray [num_predictions x num_joints x 2(x,y)]
    :param annotations: numpy.ndarray [num_predictions x num_joints x 2(x,y)]
    :param ref_length_indices: tuple of length 2, joint indices of reference distance.
    :param fallback_ref_lengths: numpy.ndarray [num_predictions] Fallback reference length for each pose.
    :return: numpy.ndarray [num_predictons x num_distances(joints)]
    """
    assert predictions.shape == annotations.shape
    assert len(ref_length_indices) == 2
    has_valid_flag = annotations.shape[2] == 3

    # Step 1: Calculate reference lengths
    ref_lengths = np.sqrt(np.sum(np.power(
        (annotations[:, ref_length_indices[0], :2] - annotations[:, ref_length_indices[1], :2]), 2), axis=1))
    if has_valid_flag:
        ref_lengths_invalid = np.logical_or(np.logical_or(ref_lengths == 0, annotations[:, ref_length_indices[0], 2] == 0),
                                            annotations[:, ref_length_indices[1], 2] == 0)
        annotations_invalid = annotations[:, :, 2] == 0
        if fallback_ref_lengths is not None:
            ref_lengths[ref_lengths_invalid] = fallback_ref_lengths[ref_lengths_invalid]
        else:
            # Set invalid ref lengths to some non-zero value for save division, will be filtered out later on
            ref_lengths[ref_lengths_invalid] = 1

    # Step 2: Calculate joint-wise distance between prediction and annotation
    distances = np.sqrt(np.sum(np.power(predictions[:, :, :2] - annotations[:, :, :2], 2), axis=2))

    # Step 3: Convert euclidean distances to normalized PCK distances
    norm_distances = distances / ref_lengths[:, np.newaxis]

    # Step 4: If possible, handle cases with invalid reference distances
    if has_valid_flag:
        norm_distances[annotations_invalid] = -1
        if fallback_ref_lengths is None:
            norm_distances[ref_lengths_invalid, :] = -1

    return norm_distances


def pck_scores_from_normalized_distances(normalized_distances):
    """
    Create plottable PCK statistics from normalized pck distances.
    Returns a flattened list of pck thresholds and a list of respective pck scores.
    :param normalized_distances: numpy.ndarray of any dimension, with a total size of N.
    Normalized distances, will be flattened.
    :return: numpy.ndarrays of sizes [N], [N]. First one are the pck thresholds, second one the pck scores.
    """
    normalized_distances_flattened = normalized_distances.flatten()
    n = normalized_distances_flattened.shape[0]
    pck_thresholds = np.sort(normalized_distances_flattened)
    pck_scores = np.arange(0, n, dtype=np.float) / n
    return pck_thresholds, pck_scores


def pck_score_at_threshold(pck_thresholds, pck_scores, applied_threshold):
    """
    Given a sorted list of PCK thresholds and assigned PCK scores, return the pck score at a specific threshold.
    :param pck_thresholds: numpy.ndarray, [N], sorted PCK thresholds (ascending).
    :param pck_scores: numpy.ndarray, [N], assigend PCK scores.
    :param applied_threshold: PCK threshold to apply.
    :return: The PCK score at the specified threshold.
    """
    score_indices = np.where(pck_thresholds < applied_threshold)[0]
    if score_indices.shape[0] > 0:
        return pck_scores[score_indices[-1]]
    else:
        return .0


if __name__ == "__main__":
    pred = [2, 5] * 20
    ann = list(range(20)) * 2
    prediction = np.array([pred]).reshape((1, 20, 2))
    annotation = np.array([ann]).reshape((1, 20, 2))
    dist1 = pck_normalized_distances(prediction, annotation)
    print(dist1)

    print(pck_normalized_distances_fast(prediction, annotation,
                                        ref_length_indices=(JumpJointOrder.l_shoulder, JumpJointOrder.r_hip)))
    # Test annotations with valid-flags
    prediction = np.concatenate([prediction, np.ones((1, 20, 1))], axis=2)
    annotation = np.concatenate([annotation, np.ones((1, 20, 1))], axis=2)

    annotation_any_joint_invalid = np.copy(annotation)
    annotation_any_joint_invalid[:, [4, 12], 2] = 0

    print(pck_normalized_distances_fast(prediction, annotation_any_joint_invalid,
                                        ref_length_indices=(JumpJointOrder.l_shoulder, JumpJointOrder.r_hip)))
    # Test annotations with invalid reference joints
    annotation_any_joint_invalid[:, [JumpJointOrder.l_shoulder], 2] = 0
    print(pck_normalized_distances_fast(prediction, annotation_any_joint_invalid,
                                        ref_length_indices=(JumpJointOrder.l_shoulder, JumpJointOrder.r_hip)))

    # Test annotations with invalid reference joints and fallback distances
    fallback_dist = np.array([16])
    print(pck_normalized_distances_fast(prediction, annotation_any_joint_invalid,
                                        ref_length_indices=(JumpJointOrder.l_shoulder, JumpJointOrder.r_hip),
                                        fallback_ref_lengths=fallback_dist))
    
    
