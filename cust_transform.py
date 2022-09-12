# This scripts contains some custom transformations defined my user
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def GSD_calculation(anno_data, camera_type='Pro2', ref_altitude=60):
    height = anno_data['altitude']
    if (camera_type == 'Pro2'):
        ref_GSD = (13.2 * ref_altitude)/(10.26*5472)
        GSD = (13.2 * height)/(10.26*5472)
    elif (camera_type == 'Air2'):
        ref_GSD = (6.4*ref_altitude)/(4.3*8000)
        GSD = (6.4*height)/(4.3*8000)
    else:
        ref_GSD = (13.2 * ref_altitude)/(10.26*5472)
        GSD = (13.2 * height)/(10.26*5472)
    return GSD, ref_GSD


def RandomVerticalFlip(image, anno_data):

    image_center = np.array(image.shape[:2])[::-1]//2
    image_center = np.hstack((image_center, image_center))

    image = cv2.flip(image, 1)
    bbox = np.asarray(anno_data['bbox'])
    bbox[:, [0, 2]] += 2*(image_center[[0, 2]] - bbox[:, [0, 2]])

    box_w = abs(bbox[:, 0] - bbox[:, 2])

    bbox[:, 0] -= box_w
    bbox[:, 2] += box_w
    anno_data['bbox'] = bbox
    return image, anno_data


def RandomHorizontalFlip(image, anno_data):
    image_center = np.array(image.shape[:2])[::-1]//2
    image_center = np.hstack((image_center, image_center))

    image = cv2.flip(image, 0)
    bbox = np.asarray(anno_data['bbox'])
    bbox[:, [1, 3]] += 2*(image_center[[1, 3]] - bbox[:, [1, 3]])

    box_w = abs(bbox[:, 1] - bbox[:, 3])

    bbox[:, 1] -= box_w
    bbox[:, 3] += box_w
    anno_data['bbox'] = bbox
    return image, anno_data


def altitude_based_scale(image, anno_data, ref_altitude=60):
    # Apply a scale normalization based on the altitude of current image
    # By default we use altitude of 60M as reference
    cur_GSD, ref_GSD = GSD_calculation(
        anno_data=anno_data, camera_type='Pro2', ref_altitude=ref_altitude)
    scale = 1.0*cur_GSD/ref_GSD
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    new_bbox = []
    for box in anno_data['bbox']:
        new_bbox.append([i*scale for i in box])
    anno_data['bbox'] = new_bbox
    return image, anno_data


def points2corners(bbox):
    bbox = np.asarray(bbox)

    width = abs((bbox[:, 2] - bbox[:, 0])).reshape(-1, 1)
    height = abs((bbox[:, 3] - bbox[:, 1])).reshape(-1, 1)

    x1 = bbox[:, 0].reshape(-1, 1)
    y1 = bbox[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bbox[:, 2].reshape(-1, 1)
    y4 = bbox[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))
    return corners


def corners2points(corners):
    bbox = []
    for cor in corners:
        bbox.append([int(i) for i in [min(cor[0], cor[2]), min(
            cor[1], cor[3]), max(cor[0], cor[2]), max(cor[1], cor[3])]])
    return bbox


def random_rotate(image, anno_data, angle = random.randint(0,180)):
    """
    Rotate the bounding box
    """
    corners = points2corners(anno_data['bbox'])
    corners = corners.reshape(-1, 2)
    corners = np.hstack(
        (corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
    (h, w) = image.shape[:2]
    (cx, cy) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T
    calculated = calculated.reshape(-1, 8)
    rotated_box = get_enclosing_box(calculated)
    anno_data['bbox'] = corners2points(rotated_box)

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))
    return image, anno_data


def get_enclosing_box(corners):
    """
    Get an enclosing box for ratated corners of a bounding box
    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


class cust_sequence(object):
    def __init__(self, augumentations, probs) -> None:
        self.augmentations = augumentations
        self.probs = probs

    def __call__(self, image, anno_data):
        if (self.augmentations == None):
            return image,anno_data
        for idx, aug in enumerate(self.augmentations):
            if (type(self.probs) == list):
                prob = self.probs[idx]
            else:
                prob = self.probs
            if random.random() < prob:
                image, anno_data = aug(image, anno_data)
        return image, anno_data


def display_image(image, anno_data):
    for box in anno_data['bbox']:
        image = cv2.rectangle(
            image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    from anno_util import readTxt
    image_dir = '/home/zt253/data/WaterfowlDataset/Processed/Bird_E/mar2019_clipped_MODOC1214_0015GSD_LINE03B0552.png'
    image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
    anno_data = readTxt(
        '/home/zt253/data/WaterfowlDataset/Processed/Bird_E/mar2019_clipped_MODOC1214_0015GSD_LINE03B0552.txt')
    image, anno_data = RandomHorizontalFlip(image, anno_data)
    print(anno_data['bbox'])
    display_image(image, anno_data)
