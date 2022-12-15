#=======================================================
#Initialize model
#=======================================================

#import library
import os
import numpy as np
import sys
# sys.path.append("./face/modules/face_detection")
from process.services.FaceRecognition.face.modules.face_detection import retinaface
from process.services.FaceRecognition.face.modules.face_recognition.inference_face_embedding import get_face_embeded
import mxnet as mx
from collections import namedtuple
import mediapipe as mp

def init_models():

    print("------------------ Loading model ----------------------")
    # gpu_id = 0 # Set GPU ID in config (os environment CUDA_VISIBLE_DEVICES)

    
    # Load model
    # if cuda:

    #     model_detection = retinaface.RetinaFace(model_face_detection_path['model'], 0, 0, 'net3')
        
    #     prefix = model_face_recognition_path['model']
    #     sym, arg, aux = mx.model.load_checkpoint(prefix, 0)
    #     # define mxnet
    #     ctx = mx.gpu(gpu_id)
    #     mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    #     mod.bind(for_training=False, data_shapes=[('data', (1,3,112,112))])
    #     mod.set_params(arg, aux)
    #     batch = namedtuple('Batch', ['data'])
    #     model_recognition = [mod, batch]
            
    #     print("Done load face_recognition model")
    # else:
    #     model_detection = retinaface.RetinaFace(model_face_detection_path['model'], 0, -1, 'net3')
        
    #     prefix = model_face_recognition_path['model']
    #     sym, arg, aux = mx.model.load_checkpoint(prefix, 0)
    #     # define mxnet
    #     ctx = mx.cpu()
    #     mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    #     mod.bind(for_training=False, data_shapes=[('data', (1,3,112,112))])
    #     mod.set_params(arg, aux)
    #     batch = namedtuple('Batch', ['data'])
    #     model_recognition = [mod, batch]
            
    #     print("Done load face_recognition model")
    
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks = True, min_detection_confidence=0.8, min_tracking_confidence=0.8)
    print("Done load face_mesh model")

    # Warm up
    
    # img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    # for i in range(2):
    #     _ = get_face_embeded(img,model_recognition)
    # print("Done warm up")
    # Warm up
    # if model_config["face_recognition"]["active"]:
    #     img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    #     img = np.transpose(img, (2, 0, 1))
    #     img = torch.from_numpy(img).unsqueeze(0).float()
    #     img.div_(255).sub_(0.5).div_(0.5)
        
    #     for i in range(2):
    #         _ = model_recognition(img)
    #     print("Done warm up")
        
    return face_mesh
