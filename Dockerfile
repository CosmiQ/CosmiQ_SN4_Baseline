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
		libncurses-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    python-dev \
    python-pip \
    python-wheel \
    python-setuptools \
    unzip \
		vim \
    wget \
    build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

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



# set up conda environment and add to $PATH
RUN conda create -n space_base python=3.6 \
                    && echo "source activate space_base" > ~/.bashrc
ENV PATH /opt/conda/envs/space_base/bin:$PATH

# install GPU version of tensorflow
RUN source activate space_base && \
    conda install -n space_base -c defaults tensorflow-gpu

# install keras with tf backend
ENV KERAS_BACKEND=tensorflow
RUN source activate space_base \
  && conda install -n space_base keras

# install various conda dependencies into the space_base environment
RUN conda install -n space_base \
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
							ncurses \
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
RUN source activate space_base && python -m ipykernel.kernelspec

# open ports for jupyterlab and tensorboard
EXPOSE 8888 6006

# switch to a temp working dir to get and install spacenetutilities V3
WORKDIR /tmp/

RUN git clone https://github.com/SpaceNetChallenge/utilities.git && cd utilities && \
    git checkout spacenetV3 && \
    source activate space_base && \
    pip install --no-cache-dir --no-dependencies -e .

RUN source activate space_base && \
	  pip install -e git+git://github.com/cosmiq/cosmiq_sn4_baseline.git@1.1.2#egg=cosmiq_sn4_baseline-1.1.2

RUN ["/bin/bash"]
