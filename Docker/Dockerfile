# syntax=docker/dockerfile:1
FROM nvidia/cudagl:10.0-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# fix Nvidia keys issue https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN   apt-key del 7fa2af80
ADD   https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
RUN   dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get  install -y ffmpeg libsm6 libxext6 python3.8  python3.8-dev python3-pip && \
    rm /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3 && \
    python3 -m pip install --upgrade pip


# Install DartPy dependenciesg
RUN apt-get install -y build-essential cmake pkg-config git && \
    apt-get install -y libeigen3-dev libassimp-dev libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev && \
    apt-get install -y libtinyxml2-dev liburdfdom-dev && \
    apt-get install -y libxi-dev libxmu-dev freeglut3-dev libopenscenegraph-dev libnlopt-dev && \
    /usr/bin/python3.6 -m pip install pytest && \
    mkdir /home/repos && cd /home/repos && git clone https://github.com/pybind/pybind11 -b 'v2.2.4' --single-branch --depth 1 && \
    cd pybind11 && mkdir build && cd build && cmake .. && make -j4 && make install && \
    cd /home/repos && git clone https://github.com/dartsim/dart.git && cd dart && git checkout tags/v6.11.1

# Install DartPy from source with new bindings
COPY resources/dart_additional_files /home/repos/dart
RUN cd /home/repos/dart && mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/ -DCMAKE_BUILD_TYPE=Release -DDART_BUILD_DARTPY=ON && \
    make -j4 dartpy && make install-dartpy

# Install Requirements
ADD docker/requirements.txt /home/requirements.txt
RUN python3 -m pip install -r /home/requirements.txt

# # Install Stable-Baselines-3
# COPY external/stable-baselines3 /home/repos/stable-baselines3
# RUN cd /home/repos/stable-baselines3 && python3 -m pip install -e .
#RUN cd /usr/local/lib/python3.8/dist-packages/ && python3 -m pip install -e /home/repos/stable-baselines3
#RUN cd /usr/local/lib/python3.8/dist-packages/ && python3 /home/repos/stable-baselines3/setup.py develop

ENV PYTHONPATH='/home/repos'
ENV WANDB_API_KEY=INSERT_KEY_HERE

WORKDIR /home
