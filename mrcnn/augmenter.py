import copy
from imgaug.augmenters import flip

from mrcnn import utils


class FliplrKeypoint(flip.Fliplr):
    """
    Flip/mirror input images horizontally.

    overrides the standard keypoint augmentation for adding support on label flip as well

    """

    def __init__(self, p=0, name=None, deterministic=False, random_state=None, config=None):
        super().__init__(p, name, deterministic, random_state)
        self.config = config

    def _augment_images(self, images, random_state, parents, hooks):
        return super()._augment_images(images, random_state, parents, hooks)

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return super()._augment_heatmaps(heatmaps, random_state, parents, hooks)

    def get_parameters(self):
        return super().get_parameters()

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        keypoints_on_image_copy = None
        keypoints_label, keypoint_flip_map = utils.get_keypoints()

        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)

        for i, keypoints_per_image in enumerate(keypoints_on_images):
            if not keypoints_per_image.keypoints:
                continue
            elif samples[i] == 1:
                width = keypoints_per_image.shape[1]
                for keypoint in keypoints_per_image.keypoints:
                    keypoint.x = (width - 1) - keypoint.x

                # skip if bbox (imgaug uses _augment_keypoints internally for bbox)
                if (len(keypoints_per_image.keypoints) % self.config.NUM_KEYPOINTS) != 0:
                    continue

                keypoints_on_image_copy = copy.deepcopy(keypoints_on_images)
                for j, _ in enumerate(range(0, len(keypoints_per_image.keypoints), self.config.NUM_KEYPOINTS)):
                    for lkp, rkp in keypoint_flip_map.items():
                        lid = keypoints_label.index(lkp) + (j * self.config.NUM_KEYPOINTS)
                        rid = keypoints_label.index(rkp) + (j * self.config.NUM_KEYPOINTS)
                        keypoints_on_image_copy[i].keypoints[rid].label = keypoints_per_image.keypoints[lid].label
                        keypoints_on_image_copy[i].keypoints[lid].label = keypoints_per_image.keypoints[rid].label

        if keypoints_on_image_copy is None:
            return keypoints_on_images

        return keypoints_on_image_copy
