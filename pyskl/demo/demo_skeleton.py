# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
import warnings
from scipy.optimize import linear_sum_assignment
from draft import *
from pyskl.apis import inference_recognizer, init_recognizer

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    def inference_detector(*args, **kwargs):
        pass

    def init_detector(*args, **kwargs):
        pass
    warnings.warn(
        'Failed to import `inference_detector` and `init_detector` from `mmdet.apis`. '
        'Make sure you can successfully import these if you want to use related features. '
    )

try:
    from mmpose.apis import inference_top_down_pose_model, inference_bottom_up_pose_model, init_pose_model, vis_pose_result
except (ImportError, ModuleNotFoundError):
    def init_pose_model(*args, **kwargs):
        pass

    def inference_top_down_pose_model(*args, **kwargs):
        pass

    def inference_bottom_up_pose_model(*args, **kwargs):
        pass

    def vis_pose_result(*args, **kwargs):
        pass

    warnings.warn(
        'Failed to import `init_pose_model`, `inference_top_down_pose_model`, `vis_pose_result` from '
        '`mmpose.apis`. Make sure you can successfully import these if you want to use related features. '
    )


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1.5
FONTCOLOR = (0, 0, 255)  # BGR, white
THICKNESS = 3
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('out_filename', help='output filename')
    parser.add_argument(
        '--config',
        default='configs/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/b.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default='work_dirs/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/b-6(Change skeleton connections & cliplen100 & numclips10)/best_top1_acc_epoch_16.pth',
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--hpe-category',
        default='topdown'
    )
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_1x_coco-person.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
                 'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/nturgbd_120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    args = parser.parse_args()
    return args


def frame_extraction(video_path, short_side):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            print('The input video size is {}'.format((w, h)))
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
            print('Rescale the input image to {}'.format((new_w, new_h)))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference_topdown(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    ret = []
    print('Performing Top-Down Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret

def pose_inference_bottomup(args, frame_paths):
    if args.pose_config == 'demo/LwOpenPose/config.py':
        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(args.pose_checkpoint, map_location='cpu')
        load_state(net, checkpoint)
        video_path = args.video
        videos = cv2.VideoCapture(video_path)
        length = videos.get(cv2.CAP_PROP_FRAME_COUNT)
        pose_tracker = naive_pose_tracker(data_frame=length)
        frame_index = 0
        i = 0
        video = list()
        aaa = 0
        new_H, new_W = None, None
        prog_bar = mmcv.ProgressBar(length)
        while True:
            ret, orig_image = videos.read()
            aaa = aaa + 1

            if orig_image is None:
                break
            if new_H is None:
                source_H, source_W, _ = orig_image.shape
                print('\nThe input video size is {}'.format((source_W, source_H)))
                new_W, new_H = mmcv.rescale_size((source_W, source_H), (480, np.Inf))
                print('The resized video size is {}'.format((new_W, new_H)))
            orig_image = cv2.resize(
                    orig_image, (new_W, new_H))
            H, W, _ = orig_image.shape
            video.append(orig_image)
            # candidate,subset=body_estimate(orig_image)

            # multi_pose,posed_index,posed_index_not= estimate_bodypose(candidate, subset)
            multi_pose = light_op.estimate_pose(net, orig_image)
            multi_pose = torch.from_numpy(multi_pose)
            multi_pose = multi_pose.unsqueeze(0)

            multi_pose[:, :, 0] = multi_pose[:, :, 0] / W  # 横坐标
            multi_pose[:, :, 1] = multi_pose[:, :, 1] / H  # 纵坐标
            multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
            multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
            multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

            multi_pose = multi_pose.numpy()
            pose_tracker.update(multi_pose, frame_index)
            prog_bar.update()
            frame_index += 1
            i = i + 1
        print('\n骨架预测完成')
        data_numpy = pose_tracker.get_skeleton_sequence()
        data_numpy = convert_kps(data_numpy, 'openpose', 'coco')  # 3*339*17*1
        a3 = time()
        data_test = data_numpy
        data_test[0] = (data_test[0] + 0.5) * new_W
        data_test[1] = (data_test[1] + 0.5) * new_H
        data_test = np.transpose(data_test, (1, 3, 2, 0))
        data_final = list(list())
        for frame in data_test:
            person_l = list()
            for person in frame:
                person_l.append({'keypoints': person})
            data_final.append(person_l)
        return data_final
    else:
        model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                args.device)
        ret = []
        print('Performing Bottom-Up Human Pose Estimation for each frame')
        prog_bar = mmcv.ProgressBar(len(frame_paths))
        for f in frame_paths:
            # Align input format
            pose = inference_bottom_up_pose_model(model, f)[0]
            ret.append(pose)
            prog_bar.update()
        return ret

def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


def pose_tracking(pose_results, max_tracks=2, thre=30):
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1], poses[j])

        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]['data'].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[])
                    new_track['track_id'] = num_tracks
                    new_track['data'] = [(idx, poses[j])]
                    tracks.append(new_track)
    if num_joints is None:
        return None, None
    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose
    return result[..., :2], result[..., 2]


def main():
    args = parse_args()
    a0 = time()
    frame_paths, original_frames = frame_extraction(args.video,
                                                    args.short_side)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    # Are we using GCN for Infernece?
    GCN_flag = 'GCN' in config.model.type
    GCN_nperson = None
    if GCN_flag:
        format_op = [op for op in config.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]
        # We will set the default value of GCN_nperson to 2, which is
        # the default arg of FormatGCNInput
        GCN_nperson = format_op.get('num_person', 2)

    model = init_recognizer(config, args.checkpoint, args.device)

    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    if args.hpe_category == 'topdown':
        a1 = time()
        # Get Human detection results
        det_results = detection_inference(args, frame_paths)
        print('\n')
        torch.cuda.empty_cache()
        a2 = time()
        pose_results = pose_inference_topdown(args, frame_paths, det_results)
        # print('\nDebug:')
        # print(pose_results)  # list(T*M*dict{'keypoints': np.ndarray(17, 3), 'bbox': np.ndarray(5)})
        # print('\n')
        torch.cuda.empty_cache()
        a3 = time()
    else:
        a1 = time()
        a2 = time()
        pose_results = pose_inference_bottomup(args, frame_paths)
        torch.cuda.empty_cache()
        a3 = time()
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

    if GCN_flag:
        # We will keep at most `GCN_nperson` persons per frame.
        tracking_inputs = [[pose['keypoints'] for pose in poses] for poses in pose_results]
        keypoint, keypoint_score = pose_tracking(tracking_inputs, max_tracks=GCN_nperson)
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score
    else:
        num_person = max([len(x) for x in pose_results])
        # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
        num_keypoint = 17
        keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                            dtype=np.float16)
        keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                  dtype=np.float16)
        for i, poses in enumerate(pose_results):
            for j, pose in enumerate(poses):
                pose = pose['keypoints']
                keypoint[j, i] = pose[:, :2]
                keypoint_score[j, i] = pose[:, 2]
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score

    if fake_anno['keypoint'] is None:
        action_label = ''
    else:
        results = inference_recognizer(model, fake_anno)
        action_label = ['', '', '', '', '']
        for i in range(5):
            action_label[i] = label_map[results[i][0]] + ':' + str(round(results[i][1], 2)) + ';'
        # action_label = ''.join(action_label)

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 args.device)
    # print(pose_results)
    # print(len(pose_results), len(pose_results[0]), len(pose_results[0][0]), len(pose_results[0][0]['keypoints']))
    vis_frames = [
        vis_pose_result(pose_model, frame_paths[i], pose_results[i])
        for i in range(num_frame)
    ]
    vis_frames2 = []
    for frame in vis_frames:
        shape = frame.shape[:2]
        frame2 = cv2.resize(frame, (3 * shape[1], 3 * shape[0]))
        vis_frames2.append(frame2)
    for frame in vis_frames2:
        for i in range(5):
            cv2.putText(frame, action_label[i], (10, 32+38*i), FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames2], fps=24)
    vid.write_videofile(args.out_filename, fps=24, remove_temp=True)
    a4 = time()
    print('\n')
    print(f'Frame Extraction Time:{a1-a0:.2f}, FPS:{num_frame/(a1-a0):.2f}')
    print(f'Human Detection Time:{a2-a1:.2f}, FPS:{num_frame/(a2-a1) if a2-a1!= 0 else 0:.2f}')
    print(f'Human Pose Estimation Time:{a3-a2:.2f}, FPS:{num_frame/(a3-a2):.2f}')
    print(f'Action Recognition Time:{a4-a3:.2f}, FPS:{num_frame/(a4-a3):.2f}')
    print(f'Total Inference Time:{a4 - a0:.2f}, FPS:{num_frame/(a4-a0):.2f}')
    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()
