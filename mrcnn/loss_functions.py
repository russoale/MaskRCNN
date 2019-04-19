import keras.layers as KL
import tensorflow as tf
from keras import backend as K

from mrcnn.misc_functions import batch_pack_graph


############################################################
#  Loss Functions
############################################################


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks, input_gt_masks_train):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    input_gt_masks_train: A float32 tensor of values 0 or 1
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss * input_gt_masks_train


def keypoint_weight_loss_graph(target_keypoint_weight, pred_class, target_class_ids):
    """Loss for Mask class R-CNN whether key points are in picture.

        target_mask_class: [batch, num_rois, 17(number of keypoints)]
        pred_class: [batch, num_rois, num_classes, 2]
        target_class_ids: [batch, num_rois]. Integer class IDs.
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_mask_class = tf.cast(target_keypoint_weight, tf.int64)
    target_class_ids = K.reshape(target_class_ids, (-1,))
    pred_class = K.reshape(pred_class, (-1, 17, K.int_shape(pred_class)[3]))
    target_mask_class = tf.cast(K.reshape(target_mask_class, (-1, 17)), tf.int64)

    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]

    # Gather the positive classes (predicted and true) that contribute to loss
    target_class = tf.gather(target_mask_class, positive_roi_ix)
    pred_class = tf.gather(pred_class, positive_roi_ix)

    # Loss
    loss = K.switch(tf.size(target_class) > 0,
                    lambda: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class, logits=pred_class),
                    lambda: tf.constant(0.0))
    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_mean(loss)
    return loss


def test_keypoint_mrcnn_mask_loss_graph(target_keypoints, target_keypoint_weights, target_class_ids,
                                        pred_keypoint_logits, mask_shape=[56, 56], number_point=17):
    """
    This function is just use for inspecting the keypoint_mrcnn_mask_loss_graph
    target_keypoints: [batch, num_rois, num_keypoints].
        A int32 tensor of values between[0, 56*56). Uses zero padding to fill array.
    keypoint_weight:[batch, num_person, num_keypoint]
        0: not visible for the coressponding roi
        1: visible for the coressponding roi
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_keypoints_logit: [batch, proposals, num_keypoints, height*width] float32 tensor
                with values from 0 to 1.
       """

    # Reshape for simplicity. Merge first two dimensions into one.
    pred_keypoint_logits = KL.Lambda(lambda x: x * 1, name="pred_keypoint")(pred_keypoint_logits)
    target_keypoints = KL.Lambda(lambda x: x * 1, name="target_keypoint")(target_keypoints)
    target_class_ids = KL.Lambda(lambda x: K.reshape(x, (-1,)), name="target_class_ids_reshape")(target_class_ids)

    target_keypoints = KL.Lambda(lambda x: K.reshape(x, (-1, number_point)), name="target_keypoint_reshape")(
        target_keypoints)

    # reshape target_keypoint_weights to [N, 17]
    target_keypoint_weights = KL.Lambda(lambda x: K.reshape(x, (-1, number_point)),
                                        name="target_keypoint_weights_reshape")(target_keypoint_weights)

    # reshape pred_keypoint_masks to [N, 17, 56*56]
    pred_keypoints_logits = KL.Lambda(lambda x: K.reshape(x, (-1, number_point, mask_shape[0] * mask_shape[1])),
                                      name="pred_keypoint_reshape")(pred_keypoint_logits)

    # Only positive person ROIs contribute to the loss. And only
    # the people specific mask of each ROI.
    # positive_people_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_people_ix = KL.Lambda(lambda x: tf.where(x > 0)[:, 0], name="positive_people_ix")(target_class_ids)

    positive_people_ids = tf.cast(
        tf.gather(target_class_ids, positive_people_ix), tf.int64)

    # Gather the keypoint masks (predicted and true) that contribute to loss
    # shape: [N_positive, 17]
    positive_target_keypoints = KL.Lambda(lambda x: tf.gather(x[0], tf.cast(x[1], tf.int64)),
                                          name="positive_target_keypoints")([target_keypoints, positive_people_ix])
    # positive_target_keypoint_masks = tf.gather(target_keypoint_masks, positive_people_ix)

    # positive target_keypoint_weights to[N_positive, 17]
    positive_keypoint_weights = KL.Lambda(lambda x: tf.cast(tf.gather(x[0], tf.cast(x[1], tf.int64)), tf.int64),
                                          name="positive_keypoint_weights")(
        [target_keypoint_weights, positive_people_ix])
    # positive target_keypoint_weights to[N_positive, 17, 56*56]
    positive_pred_keypoints_logits = KL.Lambda(lambda x: tf.gather(x[0], tf.cast(x[1], tf.int64)),
                                               name="positive_pred_keypoint_masks")(
        [pred_keypoints_logits, positive_people_ix])

    positive_target_keypoints = tf.cast(positive_target_keypoints, tf.int32)
    loss = KL.Lambda(lambda x:
                     K.switch(tf.size(x[0]) > 0,
                              lambda: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(x[0], tf.int32),
                                                                                     logits=x[1]),
                              lambda: tf.constant(0.0)), name="soft_loss")(
        [positive_target_keypoints, positive_pred_keypoints_logits])

    loss = KL.Lambda(lambda x: x * tf.cast(positive_keypoint_weights, tf.float32), name="positive_loss")(loss)
    num_valid = KL.Lambda(lambda x: tf.reduce_sum(tf.cast(x, tf.float32)), name="num_valid")(positive_keypoint_weights)
    loss = KL.Lambda(lambda x:
                     K.switch(x[1] > 0,
                              lambda: tf.reduce_sum(x[0]) / x[1],
                              lambda: tf.constant(0.0)
                              ), name="keypoint_loss")([loss, num_valid])
    return loss


def mrcnn_keypoint_loss_graph(target_keypoints, target_keypoint_weights, target_class_ids, pred_keypoints_logit,
                              weight_loss=True, mask_shape=[56, 56], number_point=17):
    """Mask softmax cross-entropy loss for the keypoint head.

    target_keypoints: [batch, num_rois, num_keypoints].
        A int32 tensor of values between[0, 56*56). Uses zero padding to fill array.
    keypoint_weight:[num_person, num_keypoint]
        0: not visible for the coressponding roi
        1: visible for the coressponding roi
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_keypoints_logit: [batch, proposals, num_keypoints, height*width] float32 tensor
                with values from 0 to 1.
    """

    # Reshape for simplicity. Merge first two dimensions into one.
    # shape:[N]
    target_class_ids = K.reshape(target_class_ids, (-1,))
    # Only positive person ROIs contribute to the loss. And only
    # the people specific mask of each ROI.
    positive_people_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_people_ids = tf.cast(
        tf.gather(target_class_ids, positive_people_ix), tf.int64)

    ###Step 1 Get the positive target and predict keypoint masks
    # reshape target_keypoint_weights to [N, num_keypoints]
    target_keypoint_weights = K.reshape(target_keypoint_weights, (-1, number_point))
    # reshape target_keypoint_masks to [N, 17]
    target_keypoints = K.reshape(target_keypoints, (
        -1, number_point))

    # reshape pred_keypoint_masks to [N, 17, 56*56]
    pred_keypoints_logit = K.reshape(pred_keypoints_logit,
                                     (-1, number_point, mask_shape[0] * mask_shape[1]))

    # Gather the keypoint masks (target and predict) that contribute to loss
    # shape: [N_positive, 17]
    positive_target_keypoints = tf.cast(tf.gather(target_keypoints, positive_people_ix), tf.int32)
    # shape: [N_positive,17, 56*56]
    positive_pred_keypoints_logit = tf.gather(pred_keypoints_logit, positive_people_ix)
    # positive target_keypoint_weights to[N_positive, 17]
    positive_keypoint_weights = tf.cast(
        tf.gather(target_keypoint_weights, positive_people_ix), tf.float32)

    loss = K.switch(tf.size(positive_target_keypoints) > 0,
                    lambda: tf.nn.sparse_softmax_cross_entropy_with_logits(logits=positive_pred_keypoints_logit,
                                                                           labels=positive_target_keypoints),
                    lambda: tf.constant(0.0))
    loss = loss * positive_keypoint_weights

    if (weight_loss):
        loss = K.switch(tf.reduce_sum(positive_keypoint_weights) > 0,
                        lambda: tf.reduce_sum(loss) / tf.reduce_sum(positive_keypoint_weights),
                        lambda: tf.constant(0.0)
                        )
    else:
        loss = K.mean(loss)
    return loss
