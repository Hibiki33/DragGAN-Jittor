import os
import os.path as osp
import random

import jittor as jt
from jittor.models.inception import Inception3
import numpy as np
from PIL import Image
from scipy.linalg import sqrtm
import dlib
import json

from viz.renderer import Renderer
from utils import EasyDict

predictor_types = {
    'shape_predictor_5_face_landmarks.dat' : 5,
    'shape_predictor_68_face_landmarks.dat' : 68,
    'shape_predictor_194_face_landmarks.dat' : 194,
}
predictor_type = 'shape_predictor_5_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_type)

def detect_face_landmarks(image, landmark_num=5):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    faces = detector(image, 1)
    
    if len(faces) != 1:
        return None

    face = faces[0]
    _landmarks = predictor(image, face)

    if _landmarks.num_parts != landmark_num:
        return None

    landmarks = []
    for i in range(landmark_num):
        x, y = _landmarks.part(i).x, _landmarks.part(i).y
        landmarks.append([x, y])

    return landmarks

device = 'cuda'
cache_dir = './weights'
output_dir = './output'
valid_checkpoints_dict = {
    f.split('/')[-1].split('.')[0]: osp.join(cache_dir, f)
    for f in os.listdir(cache_dir)
    if (f.endswith('pkl') and osp.exists(osp.join(cache_dir, f)))
}
print('Valid checkpoint file:')
print(valid_checkpoints_dict)
print()

init_pkl = 'jt_stylegan3_ffhq_weights_t'

def eval_drag(save_path, index):
    seed0, seed1 = random.randint(0, 10000), random.randint(0, 10000)

    global_state = {
        "images": {
            # image_orig: the original image, change with seed/model is changed
            # image_raw: image with mask and points, change durning optimization
            # image_show: image showed on screen
        },
        "temporal_params": {
            "stop" : False,
        },
        'mask': None,  # mask for visualization, 1 for editing and 0 for unchange
        'last_mask': None,  # last edited mask
        'show_mask': True,  # add button
        "generator_params": EasyDict(),
        "params": {
            "seed": seed0,
            "motion_lambda": 20,
            "r1_in_pixels": 3,
            "r2_in_pixels": 12,
            "magnitude_direction_in_pixels": 1.0,
            "latent_space": "w+",
            "trunc_psi": 0.7,
            "trunc_cutoff": None,
            "lr": 0.001,
        },
        "device": device,
        "draw_interval": 1,
        "renderer": Renderer(disable_timing=True),
        "points": [],
        "curr_point": None,
        "curr_type_point": "start",
        'editing_state': 'add_points',
        'pretrained_weight': init_pkl
    }

    global_state['renderer'].init_network(
        global_state['generator_params'],  # res
        valid_checkpoints_dict[global_state['pretrained_weight']],  # pkl
        global_state['params']['seed'],  # w0_seed,
        None,  # w_load
        global_state['params']['latent_space'] == 'w+',  # w_plus
        'const',
        global_state['params']['trunc_psi'],  # trunc_psi,
        global_state['params']['trunc_cutoff'],  # trunc_cutoff,
        None,  # input_transform
        global_state['params']['lr']  # lr,
    )

    global_state['renderer']._render_drag_impl(
        global_state['generator_params'],
        is_drag=False,
        to_pil=True
    )

    target_image = global_state['generator_params'].image
    global_state['mask'] = np.ones((target_image.size[1], target_image.size[0]), dtype=np.uint8)
    targets = detect_face_landmarks(target_image, landmark_num=predictor_types[predictor_type])
    if targets is None:
        print('Failed to detect landmarks in the target image')
        return None, seed0, seed1

    global_state['params']['seed'] = seed1
    global_state['renderer'].init_network(
        global_state['generator_params'],  # res
        valid_checkpoints_dict[global_state['pretrained_weight']],  # pkl
        global_state['params']['seed'],  # w0_seed,
        None,  # w_load
        global_state['params']['latent_space'] == 'w+',  # w_plus
        'const',
        global_state['params']['trunc_psi'],  # trunc_psi,
        global_state['params']['trunc_cutoff'],  # trunc_cutoff,
        None,  # input_transform
        global_state['params']['lr']  # lr,
    )

    global_state['renderer']._render_drag_impl(
        global_state['generator_params'],
        is_drag=False,
        to_pil=True
    )

    source_image = global_state['generator_params'].image
    points = detect_face_landmarks(source_image, landmark_num=predictor_types[predictor_type])
    if points is None:
        print('Failed to detect landmarks in the source image')
        return None, seed0, seed1

    print('Source image landmarks:')
    print(points)
    print('Target image landmarks:')
    print(targets)

    mask = jt.array(global_state['mask']).float32()
    drag_mask = 1 - mask

    step_idx = 0
    while (not global_state["temporal_params"]["stop"]) and step_idx < 300:

        global_state['renderer']._render_drag_impl(
            global_state['generator_params'],
            points,  # point
            targets,  # target
            drag_mask,  # mask,
            global_state['params']['motion_lambda'],  # lambda_mask
            reg=0,
            feature_idx=5,  # NOTE: do not support change for now
            r1=global_state['params']['r1_in_pixels'],  # r1
            r2=global_state['params']['r2_in_pixels'],  # r2
            trunc_psi=global_state['params']['trunc_psi'],
            is_drag=True,
            to_pil=True
        )

        step_idx += 1
        points = global_state['renderer'].points

    drag_image = global_state['generator_params'].image

    # 1. mean distance
    final_points = detect_face_landmarks(drag_image, landmark_num=predictor_types[predictor_type])
    if final_points is None:
        print('Failed to detect landmarks in the final image')
        return None, seed0, seed1
    print('Final image landmarks:')
    print(final_points)
    md = np.mean(np.linalg.norm(np.array(final_points) - np.array(targets), axis=1))
    print(f'Mean distance: {md}')
    print()

    # 2. save source_image and drag_image for fid calculation
    source_image.save(osp.join(save_path, 'src', str(index) + '.png'))
    target_image.save(osp.join(save_path, 'tgt', str(index) + '.png'))
    drag_image.save(osp.join(save_path, 'drag', str(index) + '.png'))

    return md, seed0, seed1

def eval_fid(all_path, all_num):

    gt = []
    gen = []
    for i in range(all_num):
        gt.append(np.array(Image.open(osp.join(all_path, 'src', str(i) + '.png'))))
        gen.append(np.array(Image.open(osp.join(all_path, 'drag', str(i) + '.png'))))

    images1 = jt.array(np.array(gt).transpose(0, 3, 1, 2))
    images2 = jt.array(np.array(gen).transpose(0, 3, 1, 2))

    images1 = jt.nn.interpolate(images1, size=(299, 299))
    images2 = jt.nn.interpolate(images2, size=(299, 299))

    preprocess = jt.transform.Compose([
        jt.transform.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    images1 = preprocess(images1)
    images2 = preprocess(images2)

    model = Inception3(init_weights=True)
    model.eval()

    def get_features(images):
        with jt.no_grad():
            features = model(images)
            features = features[:, :-1]
            features = features.numpy()
        # Flatten features if necessary
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        return features

    features1 = get_features(images1)
    features2 = get_features(images2)

    # Debug prints for features
    # print("features1 shape:", features1.shape)
    # print("features2 shape:", features2.shape)

    if len(features1.shape) != 2 or len(features2.shape) != 2:
        raise ValueError("Features should be 2D arrays. Check model output.")

    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    # Debug prints for means and covariances
    # print("mu1 shape:", mu1.shape)
    # print("sigma1 shape:", sigma1.shape)
    # print("mu2 shape:", mu2.shape)
    # print("sigma2 shape:", sigma2.shape)

    diff = mu1 - mu2

    # Check if sigma1 or sigma2 are empty or not 2D
    if sigma1.size == 0 or sigma2.size == 0 or sigma1.ndim != 2 or sigma2.ndim != 2:
        raise ValueError("Covariance matrices are not 2D or are empty.")
    
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid

if __name__ == "__main__":

    test_case_num = 100

    with open("eval_drag.json", "w") as fp:
        result = {}
        save_path = osp.join(output_dir)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(osp.join(save_path, 'src'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'drag'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'tgt'), exist_ok=True)

        i = 0
        while i < test_case_num:
            md, s0, s1 = eval_drag(save_path, i)
            if md is None:
                continue
            result[i] = {
                "mean_distance": md,
                "target_seed": s0,
                "source_seed": s1,
            }
            i += 1

        sum_md = 0
        for i in range(test_case_num):
            sum_md += result[i]["mean_distance"]
        result["mean_distance"] = sum_md / test_case_num
        print(f'Mean distance: {result["mean_distance"]}')

        fid = eval_fid(output_dir, test_case_num)
        result["frechet_inception_distance"] = fid
        print(f'Frechet Inception Distance: {fid}')

        json.dump(result, fp, indent=4)
