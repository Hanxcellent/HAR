# show_pkl.py
import pickle

path = 'ntu120_hrnet.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径


data = pickle.load(path)

print(data)
print(len(data))
