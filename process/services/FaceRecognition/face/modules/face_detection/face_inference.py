import numpy as np
import os
import sys
import cv2
from face.modules.face_detection.utils.alignment import get_reference_facial_points, warp_and_crop_face
from face.modules.face_detection.utils.alignment_new import alignment_procedure
import imutils


def create_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


def process(img, facial_5_points, output_size):

    facial_points = np.array(facial_5_points)

    default_square = False
    inner_padding_factor = 0.5
    outer_padding = (0, 0)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    dst_img, dst_pts = warp_and_crop_face(img, facial_points, reference_pts=reference_5pts, crop_size=output_size)
    
    return dst_img, dst_pts


def pad_image(im, desired_size=192):
    """[summary]
    Args:
        im ([cv2 image]): [input image]
        desired_size (int, optional): [description]. Defaults to 64.
    Returns:
        [cv2 image]: [resized image]
    """
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im


def get_face_area(img, detector, threshold, scales = [640, 1200]):
    # print(img)
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    
    im_scale = float(target_size) / float(im_size_min)
    
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]
    flip = False
    faces, landmarks = detector.detect(img,
                                    threshold,
                                    scales=scales,
                                    do_flip=flip)
    crop_faces = []
    orginal_crops = []
    landmarks_new = []
    for i in range(len(landmarks)):

        bboxes = faces[i].astype(np.int)
        crop_face_img = img[bboxes[1]:bboxes[3],bboxes[0]:bboxes[2]]
        delta_x = bboxes[2]-bboxes[0]
        org_crop = img[max(bboxes[1]-int(delta_x//6),0):min(bboxes[3]+int(delta_x//6),img.shape[0]),max(bboxes[0]-int(delta_x//6),0):min(bboxes[2]+int(delta_x//6),img.shape[1])]
        orginal_crops.append(np.array(org_crop))

        landmark5 = landmarks[i].astype(np.int)
        if landmarks is not None:
            crop_face_img = alignment_procedure(crop_face_img, landmark5[0], landmark5[1], landmark5[2])

        crop_face_img = pad_image(crop_face_img, 112)

        crop_faces.append(crop_face_img)


    return crop_faces, faces, landmarks, landmarks_new, orginal_crops