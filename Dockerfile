FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
RUN apt-get -y update
RUN apt-get -y install g++

RUN pip install scikit-learn matplotlib opencv-python numpy mmsegmentation==0.17.0 mmcv-full==1.3.9 ftfy mmdet==2.14.0 regex
RUN apt-get install ffmpeg libsm6 libxext6 -y 