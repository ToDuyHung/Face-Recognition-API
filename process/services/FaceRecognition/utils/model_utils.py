import numpy as np
import os
import sys
import cv2
import imutils
# import sys

# sys.path.append("./face/modules/face_detection")
from process.services.FaceRecognition.face.modules.face_detection import retinaface
import mediapipe as mp
import mxnet as mx
from collections import namedtuple
from PIL import Image
import math

from support_class import BaseServiceSingleton
from config.config import Config
from common.common_keys import *
from config.constant import *


class FaceDetectionModel(BaseServiceSingleton):   
    def __init__(self, config: Config = None):
        super(FaceDetectionModel, self).__init__(config=config)
        self.model_detection = self.load_detection_model()
        self.face_alignment = FaceAlignment(config=config)

    def load_detection_model(self):

        if self.config.device:

            model_detection = retinaface.RetinaFace(self.config.model_face_detection_path, 0, 0, 'net3')        
        else:

            model_detection = retinaface.RetinaFace(self.config.model_face_detection_path, 0, -1, 'net3')
            
                        
        return model_detection
    
    def resize_img(self, im, desired_size=400):
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

        return im

    def get_face_area(self, img, threshold, scales = [640, 1200]):
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
        faces, landmarks = self.model_detection.detect(img,
                                        threshold,
                                        scales=scales,
                                        do_flip=flip)
        crop_faces = []
        orginal_crops = []
        landmarks_new = []
        for i in range(len(faces)):

            bboxes = faces[i].astype(np.int)
            crop_face_img = img[bboxes[1]:bboxes[3],bboxes[0]:bboxes[2]]
            delta_x = bboxes[2]-bboxes[0]
            org_crop = img[max(bboxes[1]-int(delta_x//6),0):min(bboxes[3]+int(delta_x//6),img.shape[0]),max(bboxes[0]-int(delta_x//6),0):min(bboxes[2]+int(delta_x//6),img.shape[1])]
            orginal_crops.append(np.array(org_crop))
            org_crop = self.resize_img(org_crop)
            crop_face_img,_ = self.face_alignment.image_preprocessing(org_crop)
            # landmark5 = landmarks[i].astype(np.int)
            # if landmarks is not None:
            #     crop_face_img = alignment_procedure(crop_face_img, landmark5[0], landmark5[1], landmark5[2])

            # crop_face_img = self.resize_img(crop_face_img, 112)
            # cv2.imwrite(f"im_{i}.jpg",org_crop)
            crop_faces.append(crop_face_img)


        return crop_faces, faces, landmarks, landmarks_new, orginal_crops


class FaceAlignment(BaseServiceSingleton):   
    def __init__(self, config: Config = None):
        super(FaceAlignment, self).__init__(config=config)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.model_face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks = True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def temperature_adjust(self, image, temper):
        tmp_image = image.copy()
        tmp_image = Image.fromarray(tmp_image)

        r, g, b = KELVIN_TABLE[temper]
        matrix = ( r / 255.0, 0.0, 0.0, 0.0,
                0.0, g / 255.0, 0.0, 0.0,
                0.0, 0.0, b / 255.0, 0.0 )
        tmp_image = tmp_image.convert('RGB', matrix)
        return np.array(tmp_image)

    def process_gamma(self, img):
        YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        Y = YCrCb[:,:,0]
        # Determine whether image is bright or dimmed
        threshold = 0.2
        exp_in = 105 # Expected global average intensity
        M, N = img.shape[:2]
        mean_in = np.sum(Y)/(M*N)
        t = (mean_in - exp_in)/ exp_in

        # Process image for gamma correction
        img_output = None
        if t < -threshold: # Dimmed Image
            img_output = self.adjust_gamma(img, 1.5)
        elif t > threshold:
            img_output = self.adjust_gamma(img, 0.7)
        else:
            img_output = self.adjust_gamma(img, 1.1)

        return img_output

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)


    def unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    def color_correction(
            self,
            img,
            shadow_amount_percent, shadow_tone_percent, shadow_radius,
            highlight_amount_percent, highlight_tone_percent, highlight_radius,
            color_percent
    ):
        """
        Image Shadow / Highlight Correction. The same function as it in Photoshop / GIMP
        :param img: input RGB image numpy array of shape (height, width, 3)
        :param shadow_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
        :param shadow_tone_percent [0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
        :param shadow_radius [>0]: Controls the size of the local neighborhood around each pixel
        :param highlight_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
        :param highlight_tone_percent [0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
        :param highlight_radius [>0]: Controls the size of the local neighborhood around each pixel
        :param color_percent [-1.0 ~ 1.0]:
        :return:
        """
        shadow_tone = shadow_tone_percent * 255
        highlight_tone = 255 - highlight_tone_percent * 255
        
        shadow_gain = 1 + shadow_amount_percent * 6
        highlight_gain = 1 + highlight_amount_percent * 6

        # extract RGB channel
        height, width = img.shape[:2]
        img = img.astype(np.float)
        img_R, img_G, img_B = img[..., 2].reshape(-1), img[..., 1].reshape(-1), img[..., 0].reshape(-1)

        # The entire correction process is carried out in YUV space,
        # adjust highlights/shadows in Y space, and adjust colors in UV space
        # convert to Y channel (grey intensity) and UV channel (color)
        img_Y = .3 * img_R + .59 * img_G + .11 * img_B
        img_U = -img_R * .168736 - img_G * .331264 + img_B * .5
        img_V = img_R * .5 - img_G * .418688 - img_B * .081312

        # extract shadow / highlight
        shadow_map = 255 - img_Y * 255 / shadow_tone
        shadow_map[np.where(img_Y >= shadow_tone)] = 0
        highlight_map = 255 - (255 - img_Y) * 255 / (255 - highlight_tone)
        highlight_map[np.where(img_Y <= highlight_tone)] = 0

        # // Gaussian blur on tone map, for smoother transition
        if shadow_amount_percent * shadow_radius > 0:
            # shadow_map = cv2.GaussianBlur(shadow_map.reshape(height, width), ksize=(shadow_radius, shadow_radius), sigmaX=0).reshape(-1)
            shadow_map = cv2.blur(shadow_map.reshape(height, width), ksize=(shadow_radius, shadow_radius)).reshape(-1)

        if highlight_amount_percent * highlight_radius > 0:
            # highlight_map = cv2.GaussianBlur(highlight_map.reshape(height, width), ksize=(highlight_radius, highlight_radius), sigmaX=0).reshape(-1)
            highlight_map = cv2.blur(highlight_map.reshape(height, width), ksize=(highlight_radius, highlight_radius)).reshape(-1)

        # Tone LUT
        t = np.arange(256)
        LUT_shadow = (1 - np.power(1 - t * (1 / 255), shadow_gain)) * 255
        LUT_shadow = np.maximum(0, np.minimum(255, np.int_(LUT_shadow + .5)))
        LUT_highlight = np.power(t * (1 / 255), highlight_gain) * 255
        LUT_highlight = np.maximum(0, np.minimum(255, np.int_(LUT_highlight + .5)))

        # adjust tone
        shadow_map = shadow_map * (1 / 255)
        highlight_map = highlight_map * (1 / 255)

        iH = (1 - shadow_map) * img_Y + shadow_map * LUT_shadow[np.int_(img_Y)]
        iH = (1 - highlight_map) * iH + highlight_map * LUT_highlight[np.int_(iH)]
        img_Y = iH

        # adjust color
        if color_percent != 0:
            # color LUT
            if color_percent > 0:
                LUT = (1 - np.sqrt(np.arange(32768)) * (1 / 128)) * color_percent + 1
            else:
                LUT = np.sqrt(np.arange(32768)) * (1 / 128) * color_percent + 1

            # adjust color saturation adaptively according to highlights/shadows
            color_gain = LUT[np.int_(img_U ** 2 + img_V ** 2 + .5)]
            w = 1 - np.minimum(2 - (shadow_map + highlight_map), 1)
            img_U = w * img_U + (1 - w) * img_U * color_gain
            img_V = w * img_V + (1 - w) * img_V * color_gain

        # re convert to RGB channel
        output_R = np.int_(img_Y + 1.402 * img_V + .5)
        output_G = np.int_(img_Y - .34414 * img_U - .71414 * img_V + .5)
        output_B = np.int_(img_Y + 1.772 * img_U + .5)

        output = np.row_stack([output_B, output_G, output_R]).T.reshape(height, width, 3)
        output = np.minimum(np.maximum(output, 0), 255).astype(np.uint8)
        return output

    def normalize_color_channel(self, image, src_img_info, patch_image):
        rmean_src, rstd_src, gmean_src, gstd_src, bmean_src, bstd_src = src_img_info
        r, g, b = cv2.split(patch_image)
        r_src, g_src, b_src = cv2.split(image)
        rmean, rstd = r[(r>=30) & (r <= 250)].mean(), r[(r>=30) & (r <= 250)].std()
        gmean, gstd = g[(g>=30) & (g <= 250)].mean(), g[(g>=30) & (g <= 250)].std()
        bmean, bstd = b[(b>=30) & (b <= 250)].mean(), b[(b>=30) & (b <= 250)].std()
        r_src[(r_src>=1) & (r_src <= 250)] = ((r_src[(r_src>=1) & (r_src <= 250)] - rmean) * (rstd_src / rstd) + rmean_src).clip(1,250)
        g_src[(g_src>=1) & (g_src <= 250)] = ((g_src[(g_src>=1) & (g_src <= 250)] - gmean) * (gstd_src / gstd) + gmean_src).clip(1,250)
        b_src[(b_src>=1) & (b_src <= 250)] = ((b_src[(b_src>=1) & (b_src <= 250)] - bmean) * (bstd_src / bstd) + bmean_src).clip(1,250)

        new_img = cv2.merge([r_src, g_src, b_src]).astype(np.uint8)
        return new_img

    def face_alignment(self, image):
        # Read image file with cv2 and process with face_mesh
        results = self.model_face_mesh.process(image)

        # Define boolean corresponding to whether or not a face was detected in the image
        face_found = bool(results.multi_face_landmarks)

        if face_found:
            # Create a copy of the image
            annotated_image = image.copy()
            landmarks = results.multi_face_landmarks[0]
            height, width = image.shape[:2]
            list_points = np.array([[int(point.x * width), int(point.y * height)] for point in landmarks.landmark])

            blank_image = np.zeros(annotated_image.shape[:2], np.uint8)
            # Get convex hull
            remapped_points = cv2.convexHull(list_points)
            cv2.fillPoly(blank_image, pts = [remapped_points], color =(255))
            # Get facemesh
            facemesh = cv2.bitwise_and(annotated_image, annotated_image, mask=blank_image)
            min_x = max(list_points[:,0].min(), 0)
            max_x = min(list_points[:,0].max(), width)
            min_y = max(list_points[:,1].min(), 0)
            max_y = min(list_points[:,1].max(), height)
            facemesh = facemesh[min_y:max_y, min_x:max_x]

            left_eye_points = []
            right_eye_points = []
            for i in range(len(list_points)):
                if (i in np.unique(list(mp.solutions.face_mesh.FACEMESH_LEFT_IRIS)).tolist()):
                    left_eye_points.append(list_points[i])
                if (i in np.unique(list(mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS)).tolist()):
                    right_eye_points.append(list_points[i])

            left_eye = np.sum(np.array(left_eye_points), axis=0) / len(left_eye_points)
            right_eye = np.sum(np.array(right_eye_points), axis=0) / len(right_eye_points)

            angle = 180 + math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / math.pi

            center_point = (annotated_image.shape[1]/2, annotated_image.shape[0]/2)
            tfm = cv2.getRotationMatrix2D(center_point, angle, 1.0)
            annotated_image = cv2.warpAffine(annotated_image, tfm, (annotated_image.shape[1], annotated_image.shape[0]))
            
            # add ones
            ones = np.ones(shape=(len(list_points), 1))
            list_points = np.hstack([list_points, ones])

            # transform points
            list_points = tfm.dot(list_points.T).T.astype(int)

            min_x = max(list_points[:,0].min(), 0)
            max_x = min(list_points[:,0].max(), width)
            min_y = max(list_points[:,1].min() - int((list_points[:,1].max() - list_points[:,1].min()) / 4), 0)
            max_y = min(list_points[:,1].max(), height)

            annotated_image = annotated_image[min_y:max_y, min_x:max_x]
            annotated_image = imutils.resize(annotated_image, height=400)
            facemesh = imutils.resize(facemesh, height=400)
            return annotated_image, facemesh, True

        return np.zeros((50,50), np.uint8), np.zeros((50,50), np.uint8), False

    def pad_image(self, im, desired_size=192):
        """[summary]
        Args:
            im ([cv2 image]): [input image]
            desired_size (int, optional): [description]. Defaults to 64.
        Returns:
            [cv2 image]: [resized image]
        """
        old_size = im.shape[:2] # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = [int(x*ratio) for x in old_size]

        # new_size should be in (width, height) format
        im = cv2.resize(im, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)

        return new_im

    def image_preprocessing(self, image, output_size=112, temperature=9000): #7000 for dataset, 9000 for test
        image_process = self.temperature_adjust(image, temperature)
        image_process = self.process_gamma(image_process)
        image_process = self.color_correction(image_process, 0.3, 0.15, 5, 0.2, 0.2, 5, 0.2)
        image_crop, facemesh, is_success = self.face_alignment(image_process)
        if (not is_success):
            return image, False

        result = self.normalize_color_channel(image_crop, [FINAL_RMEAN, FINAL_RSTD, FINAL_GMEAN, FINAL_GSTD, FINAL_BMEAN, FINAL_BSTD], facemesh)
        result = self.pad_image(result, output_size)
        result = self.unsharp_mask(result, kernel_size=(5,5), sigma=2.0, threshold=2)
        return result, True


class FaceRecognitionModel(BaseServiceSingleton):   
    def __init__(self, config: Config = None):
        super(FaceRecognitionModel, self).__init__(config=config)
        self.model_recognition = self.load_recognition_model()

    def load_recognition_model(self):

        if self.config.device:    
            prefix = self.config.model_face_recognition_path
            sym, arg, aux = mx.model.load_checkpoint(prefix, 0)
            # define mxnet
            ctx = mx.gpu(0)
            mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
            mod.bind(for_training=False, data_shapes=[('data', (1,3,112,112))])
            mod.set_params(arg, aux)
            batch = namedtuple('Batch', ['data'])
            model_recognition = [mod, batch]
        else:
            prefix = self.config.model_face_recognition_path
            sym, arg, aux = mx.model.load_checkpoint(prefix, 0)
            # define mxnet
            ctx = mx.cpu()
            mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
            mod.bind(for_training=False, data_shapes=[('data', (1,3,112,112))])
            mod.set_params(arg, aux)
            batch = namedtuple('Batch', ['data'])
            model_recognition = [mod, batch]
                
        return model_recognition

    def get_array(self, face_chip):
        face_chip = cv2.cvtColor(face_chip, cv2.COLOR_BGR2RGB)
        face_chip = face_chip.transpose(2, 0, 1)
        face_chip = face_chip[np.newaxis, :] # 4d
        array = mx.nd.array(face_chip)
        return array


    def get_face_embeded(self, img):

        mod, batch = self.model_recognition
        
        array = self.get_array(img)
        mod.forward(batch([array]))
        feat = mod.get_outputs()[0].asnumpy()

        return feat[0]


class FaceMeshModel(BaseServiceSingleton):   
    def __init__(self, config: Config = None):
        super(FaceMeshModel, self).__init__(config=config)
        self.model_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.model_face_mesh.FaceMesh(refine_landmarks = True, min_detection_confidence=0.8, min_tracking_confidence=0.8)

    def inference(self, image):

        image = imutils.resize(image, height=400)
        try:
            image = cv2.copyMakeBorder(image, 0, 0, (400-image.shape[1])//2, 400- image.shape[1]-(400-image.shape[1])//2, cv2.BORDER_CONSTANT,
            value=[0, 0, 0])
        except:
            image  =  cv2.resize(image,(400,400))

        # To improve performance
        # image.flags.writeable = False
        
        # Get the result
        results = self.face_mesh.process(image)
        
        # # To improve performance
        # image.flags.writeable = True
        
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        face_temp_3d = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_temp_3d.append([x, y, lm.z])       
                    if idx in np.unique(list(mp.solutions.face_mesh.FACEMESH_LIPS)).tolist() + [1, 4, 5, 195, 197, 6, 2, 164]:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        # x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                face_temp_3d = np.array(face_temp_3d, dtype=np.float64)
                # The camera matrix
                focal_length = 1 * img_w

                central_points = np.array([np.sum(face_temp_3d[:,0]) / len(face_temp_3d), np.sum(face_temp_3d[:,1]) / len(face_temp_3d), np.sum(face_temp_3d[:,2]) / len(face_temp_3d)])

                cam_matrix = np.array([ [focal_length, 0, central_points[0]],
                                                    [0, focal_length, central_points[1]],
                                                    [0, 0, 1]])
                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360 + 7.5
                y = angles[1] * 360
                z = angles[2] * 360
        return x, y, z