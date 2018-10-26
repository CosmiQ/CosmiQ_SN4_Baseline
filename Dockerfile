FROM nvidia/cuda:9.0-devel-ubuntu16.04
MAINTAINER nweir <nweir@iqt.org>

ENV CUDNN_VERSION 7.3.0.29
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

# prep apt-get and cudnn
RUN apt-get update && apt-get install -y --no-install-recommends \
	    apt-utils \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# install requirements
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libgdal-dev \
    libssl-dev \
    libffi-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    python-dev \
    python-pip \
    python-wheel \
    python-setuptools \
    unzip \
    wget \
    build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# install anaconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# use conda-forge instead of default channel
RUN conda update conda && \
    conda config --remove channels defaults && \
    conda config --add channels conda-forge

SHELL ["/bin/bash", "-c"]


# set up conda environment and add to $PATH
RUN conda create -n tf_keras python=3.6 \
                    && echo "source activate tf_keras" > ~/.bashrc
ENV PATH /opt/conda/envs/tf_keras/bin:$PATH

# install GPU version of tensorflow
ENV PATH /opt/conda/bin:$PATH
RUN source activate tf_keras && \
    conda install -n tf_keras -c defaults tensorflow-gpu

# install keras with tf backend
ENV KERAS_BACKEND=tensorflow
RUN source activate tf_keras \
  && conda install -n tf_keras keras

# install various conda dependencies into the tf_keras environment
RUN conda install -n tf_keras \
              awscli \
              osmnx=0.7.3 \
              affine \
              pyproj \
              pyhamcrest=1.9.0 \
              cython \
              fiona \
              h5py \
              ncurses \
              jupyter \
              jupyterlab \
              ipykernel \
              libgdal \
              matplotlib \
              numpy \
              opencv \
              pandas \
              pillow \
              pip \
              scipy \
              scikit-image \
              scikit-learn \
              shapely \
              gdal \
              rtree \
              testpath \
              tqdm \
              pandas \
              geopandas \
              rasterio

# add a jupyter kernel for the conda environment in case it's wanted
RUN source activate tf_keras && python -m ipykernel.kernelspec

# open ports for jupyterlab and tensorboard
EXPOSE 8888 6006

# switch to a temp working dir to get and install spacenetutilities V3
WORKDIR /tmp/

RUN git clone https://github.com/SpaceNetChallenge/utilities.git && cd utilities && \
    git checkout spacenetV3 && \
    source activate tf_keras && \
    pip install --no-cache-dir --no-dependencies -e .

WORKDIR /local_data/NW_Off_NadirBaseline

RUN ["/bin/bash"]
