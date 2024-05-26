import os

DET_FACTORY = ["FasterRCNNR50", "YOLOXS", "YOLOXI", "YOLOXtiny"]
POSE_FACTORY = ["LwOpenPose", "HRNet32", "LiteHRNet18", "LiteHRNet30"]
REC_FACTORY = ["STGCNXSet", "STGCNppXSet", "PoseC3DC3DXSet", "PoseC3DR3DXSet"]


def run_video(video=None, use_cpu=0):
    for item in os.scandir("demo/videos/in"):
        video_name = item.name[:-4]
        if isinstance(video, str):
            if video_name != video:
                continue
        elif isinstance(video, (list, tuple)):
            if video_name not in video:
                continue
        print(f"Processing {video_name}")
        for sets in [0, 1]:
            for rec_model in REC_FACTORY:
                if sets:
                    pose_model = "HRNet32"
                    det_model = "YOLOXS"
                else:
                    pose_model = "LiteHRNet18"
                    det_model = "YOLOXtiny"
                hpe_category = "bottomup" if pose_model == "LwOpenPose" or pose_model == "HRNet32BU" or pose_model == "HigherHRNet32" else "topdown"
                video_input = f"{video_name}.mp4"
                video_output = f"{video_name}_out" + (
                    "_" if hpe_category == "topdown" else "") + f"{det_model}_{pose_model}_{rec_model}.mp4"
                if hpe_category == "topdown":
                    shell_cmd = f"python demo/demo_skeleton.py \
                        --hpe-category {hpe_category} \
                        --det-config demo/{det_model}/config.py \
                        --det-checkpoint demo/{det_model}/model.pth \
                        --pose-config demo/{pose_model}/config.py \
                        --pose-checkpoint demo/{pose_model}/model.pth \
                        --config demo/{rec_model}/config.py \
                        --checkpoint demo/{rec_model}/model.pth \
                        demo/videos/in/{video_input} \
                        demo/videos/out/{video_output}"
                else:
                    shell_cmd = f"python demo/demo_skeleton.py \
                        --hpe-category {hpe_category} \
                        --pose-config demo/{pose_model}/config.py \
                        --pose-checkpoint demo/{pose_model}/model.pth \
                        --config demo/{rec_model}/config.py \
                        --checkpoint demo/{rec_model}/model.pth \
                        demo/videos/in/{video_input} \
                        demo/videos/out/{video_output}"

                if use_cpu:
                    shell_cmd += " --device cpu"

                os.system(shell_cmd)


sets = 1
video_name = "strech4"
use_cpu = 0
run_video(video_name)
#
#
# pose_model = "HRNet32"
# hpe_category = "bottomup" if pose_model == "LwOpenPose" or pose_model == "HRNet32BU" or pose_model == "HigherHRNet32" else "topdown"
# det_model = "YOLOXtiny" if hpe_category == "topdown" else ""
# rec_model = "STGCNXSet"
# if sets:
#     pose_model = "HRNet32"
#     det_model = "YOLOXS"
# else:
#     pose_model = "LiteHRNet18"
#     det_model = "YOLOXtiny"
#
# video_input = f"{video_name}.mp4"
# video_output = f"{video_name}_out" + (
#     "_" if hpe_category == "topdown" else "") + f"{det_model}_{pose_model}_{rec_model}.mp4"
#
# if hpe_category == "topdown":
#     shell_cmd = f"python demo/demo_skeleton.py \
#         --hpe-category {hpe_category} \
#         --det-config demo/{det_model}/config.py \
#         --det-checkpoint demo/{det_model}/model.pth \
#         --pose-config demo/{pose_model}/config.py \
#         --pose-checkpoint demo/{pose_model}/model.pth \
#         --config demo/{rec_model}/config.py \
#         --checkpoint demo/{rec_model}/model.pth \
#         demo/videos/in/{video_input} \
#         demo/videos/out/{video_output}"
# else:
#     shell_cmd = f"python demo/demo_skeleton.py \
#         --hpe-category {hpe_category} \
#         --pose-config demo/{pose_model}/config.py \
#         --pose-checkpoint demo/{pose_model}/model.pth \
#         --config demo/{rec_model}/config.py \
#         --checkpoint demo/{rec_model}/model.pth \
#         demo/videos/in/{video_input} \
#         demo/videos/out/{video_output}"
#
# if use_cpu:
#     shell_cmd += " --device cpu"
#
# os.system(shell_cmd)
