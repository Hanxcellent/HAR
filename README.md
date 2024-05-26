# 基于深度学习的人体动作识别
代码包括两个部分：dataProcess和pyskl

dataProcess用于根据训练日志(.log)和测试结果(.pkl)绘图，相关函数位于fromlog.py以及fromdoc.py

pyskl基于pyskl库(https://github.com/kennymckormick/pyskl)

## 环境配置
0. 系统环境：Ubuntu 18.04
1. 安装CUDA==1.11.0
2. 运行以下命令：
    ```shell
    git clone https://github.com/kennymckormick/pyskl.git
    cd pyskl
    conda env create -f pyskl.yaml
    conda activate pyskl
    pip install -e .
    ```

## 网络训练
1. 修改```./configs```文件夹中的训练配置文件(.py)
2. 运行以下命令以训练网络：
    ```shell
    bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
   ```
   例如，使用单个GPU训练ST-GCN网络，并在每个epoch结束时测试，则需运行：
    ```shell
    bash tools/dist_train.sh configs/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/b.py 1 --validate
   ```
   其中{other_options}的可选参数在```./tools/train.py```可以找到。
3. 训练完成后，日志与模型将保存在```./work_dirs```目录下，具体位置在训练配置文件中可以修改。

## 网络测试
1. 运行以下命令以测试网络：
   ```shell
    bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} {other_options}
   ```
   例如，使用单个GPU测试ST-GCN网络，则需运行：
   ```shell
    bash tools/dist_test.sh work_dirs/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/b-1/b.py work_dirs/stgcn/stgcn_pyskl_ntu120_xset_hrnet/b-1/best_top1_acc_epoch_16.pth 1 
   ```
2. 网络测试的输出会以pickle文件(.pkl)的形式保存在```./work_dirs```目录下。

## 动作识别系统demo
0. 系统环境：Windows10
1. 待识别视频储存于```./demo/videos/in```目录下
2. 打开./rundemo.py
3. 设置video_name，可以为单个video的名字，也可以为多个video名字组成的元组/列表如("video1","video2")。若不设置则默认对```./demo/videos/in```目录下所有视频进行识别。
4. 默认使用所有模型依次进行识别。若测试单个模型组合的效果，则注释掉8-60行，再取消61-102行的注释，然后依次设置pose_model、det_model、rec_model。可选模型在三个FACTORY中列出。
5. 运行rundemo.py，结果保存于```./demo/videos/out```目录下