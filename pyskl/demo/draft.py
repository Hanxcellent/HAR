import cv2
from processor.demo_offline import naive_pose_tracker
import numpy as np
import torch
import sys
import traceback
from collections import OrderedDict, defaultdict
from time import *
from light_openpose import light_op
from light_openpose.models.with_mobilenet import PoseEstimationWithMobileNet
from light_openpose.modules.load_state import load_state


def load_model(model, **model_args):
    # model=GCN.st_gcn.Model
    Model = import_class(model)
    model = Model(**model_args)
    return model


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                          (class_str,
                           traceback.format_exception(*sys.exc_info())))


def load_weights(model, weights):
    model = load(model, weights).cuda()
    return model


def load(model, weights_path):
    weights = torch.load(weights_path)
    weights = weights['state_dict']
    weights = OrderedDict([(k.split('module.')[-1], v.cpu()) for k, v in weights.items()])

    try:
        model.load_state_dict(weights, strict=False)
    except (KeyError, RuntimeError):
        state = model.state_dict()
        state.update(weights)
        model.load_state_dict(state)
    return model


def estimate_bodypose(candidate, subset):
    posed_index = []
    posed_index_not = []
    posed_value = []
    for i in range(18):  # i为关节点
        index = int(subset[0][i])  # 行
        if index == -1:
            posed_index_not.append(i)
            posed_value.append([0, 0, 0])
            continue
        else:
            posed_index.append(i)
            posed_value.append(candidate[index][0:3])  # x, y, score

    posed_value = np.array(posed_value)

    return posed_value, posed_index, posed_index_not


COCO_KEYPOINTS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]
OPENPOSE_18_KEYPOINTS = [
    'nose',
    'neck',
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear'
]

KEYPOINTS_FACTORY = {
    'coco': COCO_KEYPOINTS,
    'openpose': OPENPOSE_18_KEYPOINTS
}

__KEYPOINTS_MAPPING_CACHE__ = defaultdict(dict)


def convert_kps(keypoints, src, dst, keypoints_factory=None):

    if keypoints_factory is None:
        keypoints_factory = KEYPOINTS_FACTORY
    assert keypoints.ndim in {3, 4}
    if isinstance(keypoints, torch.Tensor):

        def new_array_func(shape, value, device_data, if_uint8):
            if if_uint8:
                dtype = torch.uint8
            else:
                dtype = torch.float64
            if value == 1:
                return torch.ones(
                    size=shape, dtype=dtype, device=device_data.device)
            elif value == 0:
                return torch.zeros(
                    size=shape, dtype=dtype, device=device_data.device)
            else:
                raise ValueError

        def to_type_uint8_func(data):
            return data.to(dtype=torch.uint8)

    elif isinstance(keypoints, np.ndarray):

        def new_array_func(shape, value, device_data, if_uint8):
            if if_uint8:
                dtype = np.uint8
            else:
                dtype = np.float64
            if value == 1:
                return np.ones(shape=shape, dtype=dtype)
            elif value == 0:
                return np.zeros(shape=shape, dtype=dtype)
            else:
                raise ValueError

        def to_type_uint8_func(data):
            return data.astype(np.uint8)

    src_names = keypoints_factory[src.lower()]
    dst_names = keypoints_factory[dst.lower()]
    extra_dims = keypoints.shape[:-2]
    keypoints = keypoints.reshape(-1, len(src_names), keypoints.shape[-1])

    out_keypoints = new_array_func(
        shape=(keypoints.shape[0], len(dst_names), keypoints.shape[-1]),
        value=0,
        device_data=keypoints,
        if_uint8=False)

    dst_idxs, src_idxs, _ = get_mapping(src, dst)
    out_keypoints[:, dst_idxs] = keypoints[:, src_idxs]
    out_shape = extra_dims + (len(dst_names), keypoints.shape[-1])
    out_keypoints = out_keypoints.reshape(out_shape)

    return out_keypoints


def get_mapping(src, dst, keypoints_factory=None):
    """Get mapping list from src to dst.

    Args:
        src (str): source data type from keypoints_factory.
        dst (str): destination data type from keypoints_factory.
        approximate (bool): control whether approximate mapping is allowed.
        keypoints_factory (dict, optional): A class to store the attributes.
            Defaults to keypoints_factory.

    Returns:
        list:
            [src_to_intersection_idx, dst_to_intersection_index,
             intersection_names]
    """
    if keypoints_factory is None:
        keypoints_factory = KEYPOINTS_FACTORY
    if src in __KEYPOINTS_MAPPING_CACHE__ and \
        dst in __KEYPOINTS_MAPPING_CACHE__[src]:
        return __KEYPOINTS_MAPPING_CACHE__[src][dst][:3]
    else:
        src_names = keypoints_factory[src.lower()]
        dst_names = keypoints_factory[dst.lower()]

        dst_idxs, src_idxs, intersection = [], [], []
        for dst_idx, dst_name in enumerate(dst_names):
            try:
                src_idx = src_names.index(dst_name)
            except ValueError:
                src_idx = -1
            if src_idx >= 0:
                dst_idxs.append(dst_idx)
                src_idxs.append(src_idx)
                intersection.append(dst_name)
        mapping_list = [dst_idxs, src_idxs, intersection]

        if src not in __KEYPOINTS_MAPPING_CACHE__:
            __KEYPOINTS_MAPPING_CACHE__[src] = {}
        __KEYPOINTS_MAPPING_CACHE__[src][dst] = mapping_list
        return mapping_list


