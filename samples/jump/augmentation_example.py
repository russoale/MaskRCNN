def load_image_gt_keypoints(dataset, config, image_id, augment=True, augmentation=None):
    """Load and return ground truth data for an image (image, keypoint_mask, keypoint_weight, mask, bounding boxes).

    augment: If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. NOT YET SUPPORTED!!!
        An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.

    Returns:
    image: [height, width, 3]
    iamge_meta:
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    keypoints:[num_person, num_keypoint, 3] (x, y, v) v value is as belows:
        0: not visible and without annotations
        1: not visible but with annotations
        2: visible and with annotations
    """
    # Load image and keypoints
    image = dataset.load_image(image_id)
    image_name = os.path.split(dataset.image_info[image_id]['path'])[1]
    original_shape = image.shape
    ret = dataset.load_keypoints(image_id)
    if len(ret) == 3:
        keypoints, class_ids, bbox = ret
    else:
        keypoints, _, class_ids, bbox = ret
    assert config.NUM_KEYPOINTS == keypoints.shape[1]

    if augment and config.AUG_RAND_CROP and hasattr(dataset, "random_zoom"):
        if random.randint(0, 1):
            crop = dataset.random_zoom(bbox, original_shape,
                                       min_size=config.AUG_CROP_MIN_SIZE)
            image = image[crop[1]:crop[3], crop[0]:crop[2], :]
            bbox[0][0] -= crop[0]
            bbox[0][1] -= crop[1]
            keypoints[0, :, 0] -= crop[0]
            keypoints[0, :, 1] -= crop[1]

    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    keypoints = utils.resize_keypoints(keypoints, image.shape[:2], scale, padding, crop)
    bbox = utils.resize_bbox(bbox, image.shape[:2], scale, padding, crop)

    bbox = utils.covert_bbox(bbox)
    # Random horizontal flips.

    if augment and config.AUG_RAND_ROT_ANGLE != 0:
        ang_range = config.AUG_RAND_ROT_ANGLE
        angle = (random.random() * 2 * ang_range) - ang_range
        image = ndimage.rotate(image, -angle, reshape=False)
        for k_i in range(np.shape(keypoints)[0]):
            keypoints[k_i] = utils.rotate_keypoints((image.shape[1] // 2, image.shape[0] // 2), keypoints[k_i],
                                                  angle * np.pi / 180, image.shape)
            if hasattr(dataset, "get_bbox_from_keypoints"):
                bbox[k_i] = dataset.get_bbox_from_keypoints(keypoints[k_i], image.shape, image_name)
            else:
                bbox[k_i] = utils.get_bbox_from_keypoints(keypoints[k_i], image.shape)
                # bbox[k_i] = utils.rotate_bbox((image.shape[1]//2, image.shape[0]//2), bbox, angle*np.pi/180)

    if augment and config.AUG_RAND_FLIP and hasattr(dataset, "get_keypoint_flip_map"):
        if random.randint(0, 1):
            image = np.fliplr(image)
            keypoint_names,keypoint_flip_map = dataset.get_keypoint_flip_map()
            keypoints = utils.flip_keypoints(keypoint_names, keypoint_flip_map, keypoints, image.shape[1])
            bbox = utils.flip_bbox_points(bbox, image.shape[1])

    if augment:
        # Check if any augmentation led to invalid annotations
        for n_i in range(class_ids.shape[0]):
            if class_ids[n_i] != 0:
                bbox_empty = bbox[n_i, 0] == bbox[n_i, 2] or bbox[n_i, 1] == bbox[n_i, 3]
                keypoints_empty = np.count_nonzero(keypoints[n_i, 2]) == 0
                if bbox_empty or keypoints_empty:
                    class_ids[n_i] = 0


    if augmentation:
        logging.warning("'augmentation' is not yet supported for keypoints. Use 'augment' instead.")

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    # all the class ids in the source
    # in keypoint detection task, source_class_ids = [0,1]
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, keypoints
