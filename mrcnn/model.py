"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import datetime
import multiprocessing
from collections import OrderedDict
from distutils.version import LooseVersion

import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM
import numpy as np
import os
import re
import tensorflow as tf
from keras.engine.training_generator import fit_generator

from mrcnn import utils
from mrcnn.data_formatting import compose_image_meta, parse_image_meta_graph, mold_image
from mrcnn.data_generator import DataGenerator
from mrcnn.detection_layer import DetectionLayer
from mrcnn.fpn_heads import fpn_classifier_graph, build_fpn_mask_graph, build_fpn_keypoint_graph
from mrcnn.keypoint_layer import DetectionKeypointTargetLayer
from mrcnn.keypoint_mask_layer import DetectionKeypointMaskTargetLayer
from mrcnn.load_weights import ModelCheckpointWithOptimizer
from mrcnn.loss_functions import rpn_class_loss_graph, rpn_bbox_loss_graph, mrcnn_class_loss_graph, \
    mrcnn_bbox_loss_graph, mrcnn_mask_loss_graph, mrcnn_keypoint_loss_graph
from mrcnn.mask_layer import DetectionMaskTargetLayer
from mrcnn.misc_functions import norm_boxes_graph
from mrcnn.proposal_layer import ProposalLayer
from mrcnn.resnet import resnet_graph
from mrcnn.rpn_layer import build_rpn_model
from mrcnn.utility_functions import log, compute_backbone_shapes

# Requires TensorFlow 1.12+ and Keras 2.2.0+.
assert LooseVersion(tf.__version__) >= LooseVersion("1.12")
assert LooseVersion(keras.__version__) >= LooseVersion('2.2.0')


############################################################
#  MaskRCNN Class
############################################################


class MaskRCNN:
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.training_mask = True
        self.training_keypoint = True
        if not (config.TRAINING_HEADS is None):
            self.training_mask = True if config.TRAINING_HEADS == "mask" else False
            self.training_keypoint = not self.training_mask
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.epoch = 0
        self.log_dir = ""
        self.checkpoint_path = ""
        self.set_log_dir(model_dir)
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")

        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)

            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)

            if self.training_mask:
                # 3. GT Masks (zero padded)
                # [batch, height, width, MAX_GT_INSTANCES]
                if config.USE_MINI_MASK:
                    input_gt_masks = KL.Input(
                        shape=[config.MINI_MASK_SHAPE[0],
                               config.MINI_MASK_SHAPE[1], None],
                        name="input_gt_masks", dtype=bool)
                else:
                    input_gt_masks = KL.Input(
                        shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                        name="input_gt_masks", dtype=bool)

            if self.training_keypoint:
                # 4. GT Keypoint
                # [batch, width, height, MAX_GT_INSTANCES, NUM_KEYPOINTS, (x, y, vis)]
                keypoint_scale = K.cast(K.stack([w, h, 1], axis=0), tf.float32)
                input_gt_keypoints = KL.Input(shape=[None, config.NUM_KEYPOINTS, 3])
                gt_keypoints = KL.Lambda(lambda x: x / keypoint_scale, name="gt_keypoints")(input_gt_keypoints)

        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                                train_bn=config.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
                                             stage5=True, train_bn=config.TRAIN_BN)
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        elif mode == "inference":
            anchors = input_anchors

        # RPN Model
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        name = "{}{}mrcnn".format("mask_" if self.training_mask else "",
                                  "keypoint_" if self.training_keypoint else "")

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: parse_image_meta_graph(x)["active_class_ids"]
            )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.

            if self.training_mask and self.training_keypoint:
                rois, target_class_ids, target_bbox, target_keypoint, target_keypoint_weight, target_mask = \
                    DetectionKeypointMaskTargetLayer(config, name="proposal_targets")([
                        target_rois, input_gt_class_ids, gt_boxes, gt_keypoints, input_gt_masks])
            elif self.training_keypoint:
                rois, target_class_ids, target_bbox, target_keypoint, target_keypoint_weight = \
                    DetectionKeypointTargetLayer(config, name="proposal_targets")([
                        target_rois, input_gt_class_ids, gt_boxes, gt_keypoints])
            else:
                rois, target_class_ids, target_bbox, target_mask = \
                    DetectionMaskTargetLayer(config, name="proposal_targets")([
                        target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            if self.training_mask:
                mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                                  input_image_meta,
                                                  config.MASK_POOL_SIZE,
                                                  config.NUM_CLASSES,
                                                  train_bn=config.TRAIN_BN)

            if self.training_keypoint:
                mrcnn_keypoint = build_fpn_keypoint_graph(rois, mrcnn_feature_maps, input_image_meta,
                                                          config.KEYPOINT_MASK_POOL_SIZE, config.NUM_KEYPOINTS,
                                                          train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])

            if self.training_mask:
                mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                    [target_mask, target_class_ids, mrcnn_mask])
            if self.training_keypoint:
                keypoint_loss = KL.Lambda(
                    lambda x: mrcnn_keypoint_loss_graph(*x, weight_loss=config.KEYPOINT_LOSS_WEIGHTING,
                                                        mask_shape=config.KEYPOINT_MASK_SHAPE,
                                                        number_point=config.NUM_KEYPOINTS),
                    name="mrcnn_keypoint_loss")(
                    [target_keypoint, target_keypoint_weight, target_class_ids, mrcnn_keypoint])

            # Model
            inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox,
                      input_gt_class_ids, input_gt_boxes]
            if self.training_keypoint:
                inputs.append(input_gt_keypoints)
            if self.training_mask:
                inputs.append(input_gt_masks)
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)

            outputs = [rpn_class_logits, rpn_class, rpn_bbox, mrcnn_class_logits, mrcnn_class, mrcnn_bbox, rpn_rois,
                       output_rois, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss]
            if self.training_mask:
                outputs.append(mrcnn_mask)
                outputs.append(mask_loss)
            if self.training_keypoint:
                outputs.append(mrcnn_keypoint)
                outputs.append(keypoint_loss)

            model = KM.Model(inputs, outputs, name=name)
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)

            inputs = [input_image, input_image_meta, input_anchors]
            detection = [detections, mrcnn_class, mrcnn_bbox, rpn_rois, rpn_class, rpn_bbox]
            if self.training_mask:
                mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                                  input_image_meta,
                                                  config.MASK_POOL_SIZE,
                                                  config.NUM_CLASSES,
                                                  train_bn=config.TRAIN_BN)
                detection.append(mrcnn_mask)
            if self.training_keypoint:
                keypoint_mrcnn = build_fpn_keypoint_graph(detection_boxes, mrcnn_feature_maps, input_image_meta,
                                                          config.KEYPOINT_MASK_POOL_SIZE, config.NUM_KEYPOINTS,
                                                          train_bn=config.TRAIN_BN)
                keypoint_mcrcnn_prob = KL.Activation("softmax", name="mrcnn_prob")(keypoint_mrcnn)
                detection.append(keypoint_mcrcnn_prob)

            model = KM.Model(inputs, detection, name=name)

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def compile(self, layers):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.

        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        # Set trainable layers
        if layers in layer_regex.keys():
            layers = layer_regex[layers]
        self.set_trainable(layers)

        # Optimizer object
        # Set placeholder learning rate to zero.
        # Actual learning rate is set in train().
        momentum = self.config.LEARNING_MOMENTUM
        optimizer = keras.optimizers.SGD(
            lr=0.0, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)

        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss",
                      "mrcnn_class_loss", "mrcnn_bbox_loss"]
        if self.training_keypoint:
            loss_names.append("mrcnn_keypoint_loss")
        if self.training_mask:
            loss_names.append("mrcnn_mask_loss")
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of.
            epoch, date = utils.get_epoch_and_date_from_model_path(model_path)
            if epoch is not None:
                now = date
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch.
                self.epoch = epoch - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        checkpoint_path = os.path.join(log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")

        self.log_dir = log_dir
        self.checkpoint_path = checkpoint_path

    def train(self, train_dataset, val_dataset, learning_rate, epochs,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
        custom_callbacks: Optional. Add custom callbacks to be called
            with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Data keypoint generators
        train_generator = DataGenerator(train_dataset,
                                        self.config,
                                        shuffle=True,
                                        augmentation=augmentation,
                                        batch_size=self.config.BATCH_SIZE,
                                        no_augmentation_sources=no_augmentation_sources)
        val_generator = DataGenerator(val_dataset, self.config, shuffle=True,
                                      batch_size=self.config.BATCH_SIZE)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            ModelCheckpointWithOptimizer(self.checkpoint_path,
                                         verbose=0, save_weights_only=True,
                                         save_optimizer_weights=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Set learning rate
        K.set_value(self.keras_model.optimizer.lr, learning_rate)

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count() // 2

        fit_generator(
            self.keras_model,
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
            shuffle=True
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [batch, N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [batch, N] int class IDs
        scores: [batch, N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        keypoints: [batch, N, num_keypoints, 3] (x, y, v), keypoint x, y coordinate and valid
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("windows", windows)
            log("anchors", anchors)

        # Run human pose detection
        detections, mrcnn_class, mrcnn_bbox, rois, rpn_class, rpn_bbox, mrcnn_mask, \
        mrcnn_keypoint = self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)

        if verbose:
            log("------ Predictions ----------")
            log("detections", detections)
            log("mrcnn_class", mrcnn_class)
            log("mrcnn_bbox", mrcnn_bbox)
            log("rois", rois)
            log("rpn_class", rpn_class)
            log("rpn_bbox", rpn_bbox)
            log("mrcnn_mask", mrcnn_mask)
            log("mrcnn_keypoint_prob", mrcnn_keypoint)

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_bboxes, final_class_ids, final_scores, final_masks, final_keypoints = \
                utils.unmold_mask_keypoint_detections(self.config, image.shape, molded_images[i].shape, detections[i],
                                                      windows[i], mrcnn_mask[i], mrcnn_keypoint[i])
            results.append({
                "bboxes": final_bboxes,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                "keypoints": final_keypoints
            })
        return results

    def detect_keypoint(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [batch, N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [batch, N] int class IDs
        scores: [batch, N] float probability scores for the class IDs
        keypoints: [batch, N, num_keypoints, 3] (x, y, v), keypoint x, y coordinate and valid
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert self.config.TRAINING_HEADS == "keypoint", "Check training model flag for keypoint."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("windows", windows)
            log("anchors", anchors)

        # Run human pose detection
        detections, mrcnn_class, mrcnn_bbox, rois, rpn_class, rpn_bbox, mrcnn_keypoint \
            = self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)

        if verbose:
            log("------ Predictions ----------")
            log("detections", detections)
            log("mrcnn_class", mrcnn_class)
            log("mrcnn_bbox", mrcnn_bbox)
            log("rois", rois)
            log("rpn_class", rpn_class)
            log("rpn_bbox", rpn_bbox)
            log("mrcnn_keypoint_prob", mrcnn_keypoint)

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_bboxes, final_class_ids, final_scores, final_keypoints = \
                utils.unmold_keypoint_detections(self.config, image.shape, molded_images[i].shape, detections[i],
                                                      windows[i], mrcnn_keypoint[i])
            results.append({
                "bboxes": final_bboxes,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "keypoints": final_keypoints
            })
        return results

    def detect_mask(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [batch, N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [batch, N] int class IDs
        scores: [batch, N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert self.config.TRAINING_HEADS == "mask", "Check training model flag for mask."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("windows", windows)
            log("anchors", anchors)

        # Run human pose detection
        detections, mrcnn_class, mrcnn_bbox, rois, rpn_class, rpn_bbox, mrcnn_mask, \
            = self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)

        if verbose:
            log("------ Predictions ----------")
            log("detections", detections)
            log("mrcnn_class", mrcnn_class)
            log("mrcnn_bbox", mrcnn_bbox)
            log("rois", rois)
            log("rpn_class", rpn_class)
            log("rpn_bbox", rpn_bbox)
            log("mrcnn_mask", mrcnn_mask)

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_bboxes, final_class_ids, final_scores, final_masks = \
                utils.unmold_mask_detections(detections[i], mrcnn_mask[i], image.shape, molded_images[i].shape,
                                             windows[i])

            results.append({
                "bboxes": final_bboxes,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks
            })
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Runs the detection pipeline, but expect inputs that are
        molded already. Used mostly for debugging and inspecting
        the model.

        molded_images: List of images loaded using load_image_gt()
        image_metas: image meta data, also returned by load_image_gt()

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE, \
            "Number of images must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ = \
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks = \
                utils.unmold_mask_detections(detections[i], mrcnn_mask[i],
                                             image.shape, molded_images[i].shape,
                                             window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            layer = self.find_trainable_layer(l)
            # Include layer if it has weights
            if layer.get_weights():
                layers.append(layer)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        image_metas: If provided, the images are assumed to be already
            molded (i.e. resized, padded, and normalized)

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # Run inference
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np
