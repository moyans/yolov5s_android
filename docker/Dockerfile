FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
# DEBIAN_FRONTEND这个环境变量，告知操作系统应该从哪儿获得用户输入。如果设置为”noninteractive”，你就可以直接运行命令，而无需向用户请求输入（所有操作都是非交互式的）。

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        git-lfs \
        libjpeg-dev \
        libpng-dev \
        openmpi-bin \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

# 指定PYTHON版本
ENV PYTHON_VER=3.7.6
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VER} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya

RUN pip install --upgrade pip
RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
RUN pip config set install.trusted-host mirrors.aliyun.com
RUN pip install tensorflow-gpu==2.4.0 torch==1.7.0 tensorflow-datasets


# # # install openvino
# # ENV http_proxy $HTTP_PROXY
# # ENV https_proxy $HTTP_PROXY
# # ARG DOWNLOAD_LINK=https://registrationcenter-download.intel.com/akdlm/irc_nas/17662/l_openvino_toolkit_p_2021.3.394.tgz
# # ARG INSTALL_DIR=/opt/intel/computer_vision_sdk
# # ARG TEMP_DIR=/tmp/openvino_installer
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     wget \
#     cpio \
#     sudo \
#     lsb-release && \
#     rm -rf /var/lib/apt/lists/*
# # RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
# #     wget -c $DOWNLOAD_LINK && \
# #     tar xf l_openvino_toolkit*.tgz && \
# #     cd l_openvino_toolkit* && \
# #     sed -i 's/decline/accept/g' silent.cfg && \
# #     ./install.sh -s silent.cfg && \
# #     rm -rf $TEMP_DIR
# RUN pip3 install networkx
# RUN pip3 install defusedxml

# # install requirements for yolov5
# # RUN git clone --recursive https://github.com/moyans/yolov5s_android yolov5s_android
# # COPY /yolov5s_android/yolov5/requirements.txt /yolov5/requirements.txt
# # RUN cd /yolov5 && pip3 install -r requirements.txt
# RUN apt-get update --fix-missing
# RUN apt-get install -y libgl1 libgtk-3-0 libgtk-3-dev
# RUN pip3 install pycocotools

# # install requirements for model conversion
# RUN pip3 install netron
# RUN pip3 install onnx
# RUN pip3 install onnx-simplifier
# RUN pip3 install tensorflow-datasets
# RUN pip3 install openvino2tensorflow==1.17.2
# RUN pip3 install gdown
