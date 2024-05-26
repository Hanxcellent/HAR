import glob
import os

from pyskl.smp import download_file
from pyskl.utils.visualize import Vis3DPose, Vis2DPose
from mmcv import load, dump
#%%
# Download annotations
download_file('http://download.openmmlab.com/mmaction/pyskl/demo/annotations/ntu60_samples_hrnet.pkl', 'ntu60_2d.pkl')
download_file('http://download.openmmlab.com/mmaction/pyskl/demo/annotations/ntu60_samples_3danno.pkl', 'ntu60_3d.pkl')
#%%
# Visualize 2D Skeletons without video
annotations = load('ntu60_2d.pkl')
index = 0
anno = annotations[index]
vid = Vis2DPose(anno, thre=0.2, out_shape=(540, 960), layout='coco', fps=12, video=None)
vid.ipython_display()
#%%
# Visualize 2D Skeletons with the original RGB Video
annotations = load('ntu60_2d.pkl')
index = 0
anno = annotations[index]
frame_dir = anno['frame_dir']
video_url = f"http://download.openmmlab.com/mmaction/pyskl/demo/nturgbd/{frame_dir}.avi"
download_file(video_url, frame_dir + '.avi')
vid = Vis2DPose(anno, thre=0.2, out_shape=(540, 960), layout='coco', fps=12, video=frame_dir + '.avi')
vid.ipython_display()
#%%
# Visualize 3D Skeletons
from pyskl.datasets.pipelines import PreNormalize3D
annotations = load('ntu60_3d.pkl')
index = 0
anno = annotations[index]
anno = PreNormalize3D()(anno)  # * Need Pre-Normalization before Visualization
vid = Vis3DPose(anno, layout='nturgb+d', fps=12, angle=(30, 45), fig_size=(8, 8), with_grid=False)
vid.ipython_display()
#%%
# Clean directories
os.remove('ntu60_3d.pkl')
os.remove('ntu60_2d.pkl')
for f in glob.glob("S*.avi"):
    os.remove(f)
os.remove('__temp__.mp4')