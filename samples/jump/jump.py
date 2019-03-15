import itertools
import json
import time
from collections import defaultdict

import numpy as np
import os
from numpy import pi

from mrcnn import model as modellib, dataset
from mrcnn import utils
from mrcnn.config import Config

# Root directory of the project
ROOT_DIR = os.getcwd()


class JumpConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """

    # Configuration name
    NAME = "jump"

    IMAGES_PER_GPU = 1

    GPU_COUNT = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Person and background

    BACKBONE = "resnet50"

    NUM_KEYPOINTS = 20

    KEYPOINT_MASK_SHAPE = [56, 56]

    CPM_REFINEMENT = True

    CPM_POOL_SIZES = [14, 28]

    TRAIN_ROIS_PER_IMAGE = 50

    MAX_GT_INSTANCES = 128

    RPN_TRAIN_ANCHORS_PER_IMAGE = 150

    USE_MINI_MASK = False

    DETECTION_MAX_INSTANCES = 50

    KEYPOINT_MASK_POOL_SIZE = 14

    LEARNING_RATE = 0.001

    STEPS_PER_EPOCH = 1000

    KEYPOINT_THRESHOLD = 0.005

    # Augmentation parameters
    AUG_RAND_FLIP = True

    AUG_RAND_CROP = True

    AUG_CROP_MIN_SIZE = 400

    AUG_RAND_ROT_ANGLE = 25

    PART_STR = ['head', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 'r_hand', 'l_shoulder', 'l_elbow', 'l_wrist',
                'l_hand', 'r_hip', 'r_knee', 'r_ankle', 'r_heel', 'r_toetip', 'l_hip', 'l_knee', 'l_ankle',
                'l_heel', 'l_toetip']
    # LIMBS = [0,-1,-1,5,-1,6,5,7,6,8,7,9,8,10,11,13,12,14,13,15,14,16]


Person_ID = 1


############################################################
#  Jump Class
############################################################
class JUMP:
    def __init__(self, annotation_file=None):
        """
        Constructor of helper class for reading and visualizing annotations.
        Analogous to "COCO".
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if
                                                   ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if type(catNms) == list else [catNms]
        supNms = supNms if type(supNms) == list else [supNms]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id'] in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        """
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if type(ids) == list:
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == list:
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert (type(data) == np.ndarray)
        print(data.shape)
        assert (data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i, N))
            ann += [{
                'image_id': int(data[i, 0]),
                'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                'score': data[i, 5],
                'category_id': int(data[i, 6]),
            }]
        return ann


############################################################
#  Dataset
############################################################

class JumpDataset(dataset.Dataset):
    def __init__(self, class_map=None):
        # assert task_type in ["instances", "person_keypoints"]
        # the connection between 2 close keypoints
        self._skeleton = []
        self._keypoint_names = ["head", "neck",
                                "rsho", "relb", "rwri", "rhan",
                                "lsho", "lelb", "lwri", "lhan",
                                "rhip", "rkne", "rank", "rhee", "rtoe",
                                "lhip", "lkne", "lank", "lhee", "ltoe"]
        super().__init__(class_map)

    def load_jump(self, dataset_dir, subset, class_ids=None, return_jump=False):
        """Load a subset of the jump dataset.
        dataset_dir: The root directory of the jump dataset.
        subset: What to load (train, val, minival, valminusminival)
        return_jump: If True, returns the JUMP object.
        """

        jump = JUMP("{}/annotations/keypoints_{}.json".format(dataset_dir, subset))
        image_dir = "{}/{}".format(dataset_dir, subset)

        # # Add classes
        if not class_ids:
            class_ids = sorted(jump.getCatIds())

        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(jump.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(jump.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("jump", i, jump.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "jump", image_id=i,
                path=os.path.join(image_dir, jump.imgs[i]['file_name']),
                width=jump.imgs[i]["width"],
                height=jump.imgs[i]["height"],
                annotations=jump.loadAnns(jump.getAnnIds(imgIds=[i])))

        print("Skeleton:", np.shape(self._skeleton))
        print("Keypoint names:", np.shape(self._keypoint_names))
        if return_jump:
            return jump

    @property
    def skeleton(self):
        return self._skeleton

    @property
    def keypoint_names(self):
        return self._keypoint_names

    def load_keypoints(self, image_id):
        """Load person keypoints for the given image.

        Returns:
        key_points: num_keypoints coordinates and visibility (x,y,v)  [num_person,num_keypoints,3] of num_person
        class_ids: a 1D array of class IDs of the instance masks, here is always equal to [num_person, 1]
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        # if image_info["source"] != "coco":
        #     return super(CocoDataset, self).load_mask(image_id)

        keypoints = []
        class_ids = []
        instance_masks = []
        bboxs = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = 1
            assert class_id == 1
            if class_id:
                # load keypoints
                keypoint = annotation["keypoints"]
                keypoint = np.reshape(keypoint, (-1, 3))
                bbox = annotation['bbox']

                keypoints.append(keypoint)
                class_ids.append(class_id)
                bboxs.append(bbox)

        # Pack instance masks into an array
        if class_ids:
            keypoints = np.array(keypoints, dtype=np.int32)
            class_ids = np.array(class_ids, dtype=np.int32)
            bboxs = np.array(bboxs, dtype=np.int32)
            return keypoints, class_ids, bboxs
        else:
            # Call super class to return an empty mask
            return super(JumpDataset, self).load_keypoints(image_id)

    def get_keypoint_flip_map(self):
        """Get the keypoints and their left/right flip correspondence map."""
        keypoints = self.keypoint_names
        keypoint_flip_map = {
            'rsho': 'lsho',
            'relb': 'lelb',
            'rwri': 'lwri',
            'rhan': 'lhan',
            'rhip': 'lhip',
            'rkne': 'lkne',
            'rank': 'lank',
            'rhee': 'lhee',
            'rtoe': 'ltoe',
        }
        return keypoints, keypoint_flip_map

    def random_zoom(self, bbox, shape, min_size):
        x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[0][0] + bbox[0][2], bbox[0][1] + bbox[0][3]
        if x1 > 0:
            b_x = x1 - np.random.randint(x1)
        else:
            b_x = 0
        if y1 > 0:
            b_y = y1 - np.random.randint(y1)
        else:
            b_y = 0

        if shape[1] - b_x < min_size:
            b_x = np.random.randint(shape[1] - min_size)
        if shape[0] - b_y < min_size:
            b_y = np.random.randint(shape[0] - min_size)

        if shape[1] - x2 > 0:
            bb_x = x2 + np.random.randint(shape[1] - x2)
        else:
            bb_x = x2
        if shape[0] - y2 > 0:
            bb_y = y2 + np.random.randint(shape[0] - y2)
        else:
            bb_y = y2

        return [b_x, b_y, bb_x, bb_y]

    def get_bbox_from_keypoints(self, keypoints, image_shape):
        """
        Pls, don't ask ...
        :param keypoints:
        :param image_shape:
        :param iamge_name:
        :return: Box in (y1,x1,y2,x2) format.
        """
        eps = 1e-7
        valid_keypoints = keypoints[:, 2] > 0
        x_min = np.min(keypoints[:, 0][valid_keypoints])
        y_min = np.min(keypoints[:, 1][valid_keypoints])

        x_max = np.max(keypoints[:, 0][valid_keypoints])
        y_max = np.max(keypoints[:, 1][valid_keypoints])

        headtip_x = handtip_rx = handtip_lx = elbowtip_rx = elbowtip_lx = kneetip_rx = kneetip_lx = heeltip_rx = \
            heeltip_lx = toetip_rx = toetip_lx = hiptip_lx = hiptip_rx = shouldertip_rx = shouldertip_lx = \
            ank_kneetip_rx = ank_kneetip_lx = x_min

        headtip_y = handtip_ry = handtip_ly = elbowtip_ry = elbowtip_ly = kneetip_ry = kneetip_ly = heeltip_ry = \
            heeltip_ly = toetip_ry = toetip_ly = hiptip_ly = hiptip_ry = shouldertip_ry = shouldertip_ly = \
            ank_kneetip_ry = ank_kneetip_ly = y_min

        # Head-Neck

        if keypoints[0][2] > 0 and keypoints[1][2] > 0:
            slope = np.float((keypoints[1][1] - keypoints[0][1]) / (keypoints[1][0] - keypoints[0][0] + eps))
            ang = np.abs(np.arctan(slope))
            if ang < pi / 24:
                headtip_x, headtip_y = 3 * keypoints[0][0] - 2 * keypoints[1][0], 11 * keypoints[0][1] - 10 * \
                                       keypoints[1][
                                           1]
            elif ang < pi / 12:
                headtip_x, headtip_y = 3 * keypoints[0][0] - 2 * keypoints[1][0], 7 * keypoints[0][1] - 6 * \
                                       keypoints[1][1]
            elif ang < pi / 6:
                headtip_x, headtip_y = 3 * keypoints[0][0] - 2 * keypoints[1][0], 5 * keypoints[0][1] - 4 * \
                                       keypoints[1][1]
            elif ang < pi / 4:
                headtip_x, headtip_y = 3 * keypoints[0][0] - 2 * keypoints[1][0], 3.5 * keypoints[0][1] - 2.5 * \
                                       keypoints[1][1]
            elif ang < pi / 3:
                headtip_x, headtip_y = 3 * keypoints[0][0] - 2 * keypoints[1][0], 3 * keypoints[0][1] - 2 * \
                                       keypoints[1][1]
            else:
                headtip_x, headtip_y = 3 * keypoints[0][0] - 2 * keypoints[1][0], 2.6 * keypoints[0][1] - 1.6 * \
                                       keypoints[1][1]
        # elif keypoints[0][2] == 0:
        #     y_min = 0

        # Hands
        if keypoints[4][2] > 0 and keypoints[5][2] > 0:
            handtip_rx, handtip_ry = 3 * keypoints[5][0] - 2 * keypoints[4][0], 3 * keypoints[5][1] - 2 * keypoints[4][
                1]
        # elif keypoints[5][2] == 0:
        #     y_min = 0

        if keypoints[8][2] > 0 and keypoints[9][2] > 0:
            handtip_lx, handtip_ly = 3 * keypoints[9][0] - 2 * keypoints[8][0], 3 * keypoints[9][1] - 2 * keypoints[8][
                1]
        # elif keypoints[9][2] == 0:
        #     y_min = 0

        # Shoulder-elbow
        if keypoints[2][2] > 0 and keypoints[3][2] > 0:
            elbowtip_rx, elbowtip_ry = (10 * keypoints[3][0] - 3 * keypoints[2][0]) / 7.0, (
                    10 * keypoints[3][1] - 3 * keypoints[2][1]) / 7.0
            shouldertip_rx, shouldertip_ry = (10 * keypoints[2][0] - 3 * keypoints[3][0]) / 7.0, (
                    10 * keypoints[2][1] - 3 * keypoints[3][1]) / 7.0

        if keypoints[6][2] > 0 and keypoints[7][2] > 0:
            elbowtip_lx, elbowtip_ly = (10 * keypoints[7][0] - 3 * keypoints[6][0]) / 7.0, (
                    10 * keypoints[7][1] - 3 * keypoints[6][1]) / 7.0
            shouldertip_lx, shouldertip_ly = (10 * keypoints[6][0] - 3 * keypoints[7][0]) / 7.0, (
                    10 * keypoints[6][1] - 3 * keypoints[7][1]) / 7.0

        # Hip-Knee
        if keypoints[10][2] > 0 and keypoints[11][2] > 0:
            kneetip_rx, kneetip_ry = (10 * keypoints[11][0] - 2 * keypoints[10][0]) / 8.0, (
                    10 * keypoints[11][1] - 2 * keypoints[10][1]) / 8.0
            hiptip_rx, hiptip_ry = (10 * keypoints[10][0] - 3 * keypoints[11][0]) / 7.0, (
                    10 * keypoints[10][1] - 3 * keypoints[11][1]) / 7.0

        if keypoints[15][2] > 0 and keypoints[16][2] > 0:
            kneetip_lx, kneetip_ly = (10 * keypoints[16][0] - 2 * keypoints[15][0]) / 8.0, (
                    10 * keypoints[16][1] - 2 * keypoints[15][1]) / 8.0
            hiptip_lx, hiptip_ly = (10 * keypoints[15][0] - 3 * keypoints[16][0]) / 7.0, (
                    10 * keypoints[15][1] - 3 * keypoints[16][1]) / 7.0

        # Ankle-Knee
        if keypoints[11][2] > 0 and keypoints[12][2] > 0:
            ank_kneetip_rx, ank_kneetip_ry = (10 * keypoints[11][0] - 2.5 * keypoints[12][0]) / 7.5, (
                    10 * keypoints[11][1] - 2.5 * keypoints[12][1]) / 7.5

        if keypoints[16][2] > 0 and keypoints[17][2] > 0:
            ank_kneetip_lx, ank_kneetip_ly = (10 * keypoints[16][0] - 2.5 * keypoints[17][0]) / 7.5, (
                    10 * keypoints[16][1] - 2.5 * keypoints[17][1]) / 7.5

        # Ankle-Heel
        if keypoints[12][2] > 0 and keypoints[13][2] > 0:
            slope = np.float((keypoints[13][1] - keypoints[12][1]) / (keypoints[13][0] - keypoints[12][0] + eps))
            ang = np.abs(np.arctan(slope))
            if ang < pi / 4:
                heeltip_rx, heeltip_ry = (2 * keypoints[13][0] - 1 * keypoints[12][0]), (
                        2.5 * keypoints[13][1] - 1.5 * keypoints[12][1])
            else:
                heeltip_rx, heeltip_ry = (2.5 * keypoints[13][0] - 1.5 * keypoints[12][0]), (
                        2 * keypoints[13][1] - 1 * keypoints[12][1])

        if keypoints[17][2] > 0 and keypoints[18][2] > 0:
            slope = np.float((keypoints[18][1] - keypoints[17][1]) / (keypoints[18][0] - keypoints[17][0] + eps))
            ang = np.abs(np.arctan(slope))
            if ang < pi / 4:
                heeltip_lx, heeltip_ly = (2 * keypoints[18][0] - keypoints[17][0]), (
                        2.5 * keypoints[18][1] - 1.5 * keypoints[17][1])
            else:
                heeltip_lx, heeltip_ly = (2.5 * keypoints[18][0] - 1.5 * keypoints[17][0]), (
                        2 * keypoints[18][1] - keypoints[17][1])

        # Heel-Toe
        if keypoints[13][2] > 0 and keypoints[14][2] > 0:
            slope = np.float((keypoints[13][1] - keypoints[14][1]) / (keypoints[13][0] - keypoints[14][0] + eps))
            ang = np.abs(np.arctan(slope))
            if ang < pi / 4:
                toetip_rx, toetip_ry = (10 * keypoints[14][0] - 2 * keypoints[13][0]) / 8.0, (
                        10 * keypoints[14][1] - 3 * keypoints[13][1]) / 7.0
            else:
                toetip_rx, toetip_ry = (10 * keypoints[14][0] - 3 * keypoints[13][0]) / 7.0, (
                        10 * keypoints[14][1] - 2 * keypoints[13][1]) / 8.0

        if keypoints[18][2] > 0 and keypoints[19][2] > 0:
            slope = np.float((keypoints[18][1] - keypoints[19][1]) / (keypoints[18][0] - keypoints[19][0] + eps))
            ang = np.abs(np.arctan(slope))
            if ang < pi / 4:
                toetip_lx, toetip_ly = (10 * keypoints[19][0] - 2 * keypoints[18][0]) / 8.0, (
                        10 * keypoints[19][1] - 3 * keypoints[18][1]) / 7.0
            else:
                toetip_lx, toetip_ly = (10 * keypoints[19][0] - 3 * keypoints[18][0]) / 7.0, (
                        10 * keypoints[19][1] - 2 * keypoints[18][1]) / 8.0

        xs = np.array(
            [headtip_x, handtip_rx, handtip_lx, elbowtip_lx, elbowtip_rx, kneetip_rx, kneetip_lx, heeltip_rx,
             heeltip_lx,
             toetip_rx, toetip_lx, hiptip_lx, hiptip_rx, shouldertip_rx, shouldertip_lx, ank_kneetip_rx,
             ank_kneetip_lx])
        #     x_min = min(x_min, np.min(xs[np.logical_and(xs>=0, xs<image_shape[1])]))
        #     x_max = max(x_max, np.max(xs[np.logical_and(xs>=0, xs<image_shape[1])]))
        x_min = max(0, min(x_min, np.min(xs)))
        x_max = min(image_shape[1], max(x_max, np.max(xs)))

        ys = np.array(
            [headtip_y, handtip_ry, handtip_ly, elbowtip_ly, elbowtip_ry, kneetip_ry, kneetip_ly, heeltip_ry,
             heeltip_ly,
             toetip_ry, toetip_ly, hiptip_ly, hiptip_ry, shouldertip_ry, shouldertip_ly, ank_kneetip_ry,
             ank_kneetip_ly])
        #     y_min = min(y_min, np.min(ys[np.logical_and(ys>=0, ys<image_shape[0])]))
        #     y_max = max(y_max, np.max(ys[np.logical_and(ys>=0, ys<image_shape[0])]))
        y_min = max(0, min(y_min, np.min(ys)))
        y_max = min(image_shape[0], max(y_max, np.max(ys)))

        x_top, y_top, x_bot, y_bot = int(np.floor(x_min)), int(np.floor(y_min)), int(np.ceil(x_max)), int(
            np.ceil(y_max))

        return [y_top, x_top, y_bot, x_bot]


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Jump dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on Jump dataset. 'evaluate' is not supported yet.")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/jump_dataset/",
                        help='Directory of the Jump dataset')
    parser.add_argument('--model', required=False,
                        default=None,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--continue_training', required=False,
                        default=True,
                        metavar="<True|False>",
                        help="Try to continue a previously started training, \
                            mainly by trying to recreate the optimizer state and epoch number.")
    parser.add_argument('--logs', required=True,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()
    print("Command: ", args.continue_training)
    print("Model: ", args.model)
    print("Continue Training: ", args.continue_training)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Let's support links and HOME abbreviations
    args.dataset = os.path.realpath(os.path.expanduser(args.dataset))
    args.model = os.path.realpath(os.path.expanduser(args.model))
    args.logs = os.path.realpath(os.path.expanduser(args.logs))

    # Configurations
    if args.command == "train":
        config = JumpConfig()
    else:
        class InferenceConfig(JumpConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = JumpDataset()
        dataset_train.load_jump(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = JumpDataset()
        dataset_val.load_jump(args.dataset, "val")
        dataset_val.prepare()

        print("Train Keypoints Image Count: {}".format(len(dataset_train.image_ids)))
        print("Train Keypoints Class Count: {}".format(dataset_train.num_classes))
        for i, info in enumerate(dataset_train.class_info):
            print("{:3}. {:50}".format(i, info['name']))

        print("Val Keypoints Image Count: {}".format(len(dataset_val.image_ids)))
        print("Val Keypoints Class Count: {}".format(dataset_val.num_classes))
        for i, info in enumerate(dataset_val.class_info):
            print("{:3}. {:50}".format(i, info['name']))

        # *** This is the training phase schedule ***
        lr_values = [config.LEARNING_RATE,
                     config.LEARNING_RATE,
                     config.LEARNING_RATE / 10]
        epochs_values = [10,
                         120,
                         160]
        trainable_layers = ["heads",
                            "4+",
                            "all"]

        exclude = ["mrcnn_keypoint_mask_deconv"]

        last_layers = None
        last_epoch = None
        if model_path is not None and args.continue_training:
            last_epoch, _ = model.get_epoch_and_date_from_model_path(model_path=model_path)
        weights_loaded = False
        can_load_optimizer_weights = args.continue_training

        # Run all training phases
        for i_t, (lr, epochs, layers) in enumerate(zip(lr_values, epochs_values, trainable_layers)):
            if last_epoch is not None and last_epoch >= epochs:
                # If we start with a new training phase, optimizer state can only be reused,
                # if the trainable layers did not change.
                if last_epoch == epochs:
                    if len(trainable_layers) > (i_t + 1) and layers != trainable_layers[i_t + 1]:
                        can_load_optimizer_weights = False
                # Skip this training stage as it has already been completed
                continue

            # If the trainable layers changed, we need to recompile the model
            if layers != last_layers:
                model.compile(layers=layers)

            # For the first training phase to run, load the initial weights
            if not weights_loaded and model_path is not None:
                # Load weights
                print("Loading weights from: ", model_path)
                model.load_weights(model_path, by_name=True,
                                   exclude=exclude,
                                   include_optimizer=can_load_optimizer_weights)
                weights_loaded = True
            weights_loaded = True
            if config.GPU_COUNT > 1:
                keras_layers = model.keras_model.layers
                keras_model = [keras_layer for keras_layer in keras_layers if keras_layer.name == "keypoint_mask_rcnn"][
                    0]
            else:
                keras_model = model.keras_model

            if config.CPM_REFINEMENT:
                for i in range(8):
                    weights = keras_model.get_layer(name='mrcnn_keypoint_mask_conv{}'.format(i + 1)).get_weights()
                    keras_model.get_layer(name='mrcnn_keypoint_mask_cpm1_conv{}'.format(i + 1)).set_weights(weights)

                    weights = keras_model.get_layer(name='mrcnn_keypoint_mask_bn{}'.format(i + 1)).get_weights()
                    keras_model.get_layer(name='mrcnn_keypoint_mask_cpm1_bn{}'.format(i + 1)).set_weights(weights)
            print("Training: {}".format(layers))
            model.train(dataset_train, dataset_val,
                        learning_rate=lr,
                        epochs=epochs)

            last_layers = layers

    elif args.command == "evaluate":
        print("'evaluate' is not supported yet ...")
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
