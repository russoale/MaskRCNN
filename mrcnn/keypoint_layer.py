import tensorflow as tf
from keras import engine as KE

from mrcnn import utils
from mrcnn.detection_target_layer import overlaps_graph
from mrcnn.misc_functions import trim_zeros_graph


def detection_keypoint_targets_graph(proposals, gt_class_ids, gt_boxes, gt_keypoints, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_keypoints: [MAX_GT_INSTANCES, NUM_KEYPOINTS, 3] of (x, y ,v)
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.
    Returns: Target ROIs and corresponding class IDs, bounding box shifts, keypoint label, keypoint weight
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
            Class-specific bbox refinements.
    keypoints_labels: [TRAIN_ROIS_PER_IMAGE, NUM_KEYPOINTS). Keypoint labels in [0, HEATMAP_SIZE-1]
    HEATMAP_SIZE = HEAT_MAP_WITHD * HEAT_MAP_HEIGHT

    keypoints_weights: [TRAIN_ROIS_PER_IMAGE, NUM_KEYPOINTS), 0: not visible 1: visible

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    #non_zeros:[N_box,1] true false
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_keypoints = tf.gather(gt_keypoints, tf.where(non_zeros)[:, 0], axis=0,
                         name="trim_gt_keypoints")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    crowd_keypoints = tf.gather(gt_keypoints, crowd_ix)
    crowd_masks = tf.gather(gt_masks, crowd_ix, axis=2)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)

    gt_keypoints = tf.gather(gt_keypoints, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [anchors, crowds]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine postive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks

    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_keypoints = tf.gather(gt_keypoints, roi_gt_box_assignment)
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)
    # Compute mask targets
    boxes = positive_rois
    y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
    if config.USE_MINI_MASK:
        # Transform ROI corrdinates from normalized image space
        # to normalized mini-mask space.
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                              box_ids,
                                              config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)
    masks = tf.round(masks)

    ## Transform ROI keypoints from (x,y) image space to keypoint label
    y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
    y1 = y1[:, 0]
    x1 = x1[:, 0]
    y2 = y2[:, 0]
    x2 = x2[:, 0]
    scale_x = tf.cast(config.KEYPOINT_MASK_SHAPE[1] / ((x2 - x1) * config.IMAGE_SHAPE[1]), tf.float32)
    scale_y = tf.cast(config.KEYPOINT_MASK_SHAPE[0] / ((y2 - y1) * config.IMAGE_SHAPE[0]), tf.float32)
    keypoint_lables = []
    keypoint_weights = []


    for k in range(config.NUM_KEYPOINTS):
        vis = roi_keypoints[:, k, 2] > 0
        x = tf.cast(roi_keypoints[:, k, 0], tf.float32)
        y = tf.cast(roi_keypoints[:, k, 1], tf.float32)

        # # recover from normlized corrdinates to real wordl
        x_real = (x - x1)* config.IMAGE_SHAPE[1]
        y_real = (y - y1)* config.IMAGE_SHAPE[0]
        ## transform the box size into feature map size
        x_real_map = tf.cast(x_real * scale_x+0.5, tf.int32)
        y_real_map= tf.cast(y_real*scale_y+0.5,tf.int32)
        x_boundary_bool = tf.cast((x_real_map == config.KEYPOINT_MASK_SHAPE[1]), tf.int32)
        y_boundary_bool = tf.cast((y_real_map == config.KEYPOINT_MASK_SHAPE[1]), tf.int32)
        y_real_map = y_real_map * (1 - y_boundary_bool) + y_boundary_bool * (config.KEYPOINT_MASK_SHAPE[0] - 1)
        x_real_map = x_real_map * (1 - x_boundary_bool) + x_boundary_bool * (config.KEYPOINT_MASK_SHAPE[1] - 1)

        valid_loc = tf.logical_and(
            tf.logical_and(x_real_map > 0, x_real_map < config.KEYPOINT_MASK_SHAPE[0]),
            tf.logical_and(y_real_map > 0, y_real_map < config.KEYPOINT_MASK_SHAPE[1])
        )
        valid = tf.logical_and(valid_loc, vis)
        keypoint_weights.append(valid)

        valid = tf.cast(valid, tf.int32)
        x_real_map = x_real_map * tf.cast(valid, tf.int32)
        y_real_map = y_real_map * tf.cast(valid, tf.int32)

        #calculate the keypoint label betwween[0, map_h*map_w)
        keypoint_label = y_real_map * config.KEYPOINT_MASK_SHAPE[1] + x_real_map
        keypoint_label = tf.expand_dims(keypoint_label, -1)
        keypoint_lables.append(keypoint_label)

    # shape:[N_roi, num_keypoint]
    keypoint_lables = tf.cast(tf.concat(keypoint_lables, axis=1), tf.int32)
    keypoint_weights = tf.cast(tf.stack(keypoint_weights, axis=1), tf.int32)


    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])

    keypoint_lables = tf.pad(keypoint_lables, [(0, N + P), (0, 0)])
    # keypoint_lables = tf.pad(keypoint_lables, [(0, N + P), (0, 0),(0,0)])
    keypoint_weights = tf.pad(keypoint_weights, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [(0, N + P), (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, keypoint_lables, keypoint_weights, masks



class DetectionKeypointTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,keypoint_weights
    and keypoint_masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals, Here N <= RPN_TRAIN_ANCHORS_PER_IMAGE(256)
               because of the NMS e.t.c
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_keypoints: [batch, MAX_GT_INSTANCES, NUM_KEYPOINTS, 3]
                (x, y, v)
    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    keypoint_weights and keypoint_masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinements.
    target_keypoints: [batch, TRAIN_ROIS_PER_IMAGE, NUM_KEYPOINTS)
                 Keypoint labels cropped to bbox boundaries and resized to neural
                 network output size. Maps keypoints from the half-open interval [x1, x2) on continuous image
                coordinates to the closed interval [0, HEATMAP_SIZE - 1]

    target_keypoint_weights: [batch, TRAIN_ROIS_PER_IMAGE, NUM_KEYPOINTS), bool type
                 Keypoint_weights, 0: isn't visible, 1: visilble

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionKeypointTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_keypoints = inputs[3]
        gt_masks = inputs[4]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_keypoint","target_keypoint_weight","target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_keypoints, gt_masks],
            lambda r, x, y, z, m: detection_keypoint_targets_graph(
                r, x, y, z, m,self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, 1),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE,self.config.NUM_KEYPOINTS) , # keypoint_labels
            (None, self.config.TRAIN_ROIS_PER_IMAGE,self.config.NUM_KEYPOINTS),  # keypoint_weights
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    # def compute_keypoint_mask(self, inputs, keypoin_mask=None):
    #     return [None, None, None, None, None,None]
