############################################################
#  Test
############################################################
from unittest import TestCase

import os
import random

from mrcnn import augmenter
from mrcnn import data_generator, visualize
from samples.jump import jump


class JumpTests(TestCase):

    def load_test_config(self):
        class TestConfig(jump.JumpConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
            TRAINING_HEADS = "mask"

        config = TestConfig()
        config.display()
        return config

    def load_dataset(self):
        # Validation dataset
        JUMP_DIR = os.path.abspath("/data/hdd/russales/JumpDataset/mscoco_format")
        dataset = jump.JumpDataset()
        dataset.load_jump(JUMP_DIR, "val")
        dataset.prepare()
        print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
        return dataset

    def test_load_mask(self):
        # load image and ground truth data
        config = self.load_test_config()
        dataset = self.load_dataset()
        dg = data_generator.DataGenerator(dataset, config, shuffle=True, batch_size=config.BATCH_SIZE)

        #image_id = random.choice(dataset.image_ids)
        image_id = 6

        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))

        augmentation = augmenter.FliplrKeypoint(1, config=config, dataset=dataset)

        image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_keypoints = \
            dg.load_image_gt(image_id, augmentation=augmentation, use_mini_mask=False)

        # visualize.display_keypoints(image, gt_boxes, gt_keypoints, gt_class_ids, dataset.class_names,
        #                             ax=ax, title="Original", dataset=dataset)

        visualize.display_instances(image, gt_boxes, gt_masks, gt_class_ids,
                                    dataset.class_names, title="Original")
        pass
