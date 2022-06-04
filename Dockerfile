FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV CONDA=Miniconda3-py37_4.12.0-Linux-x86_64.sh \
    EMAN2=eman2.91_sphire1.4_sparx.linux64.sh \
    PATH=/opt/conda/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive

RUN echo "now building..." && \
    apt-get update && \
    apt-get install -y apt-utils && \
    apt-get install -y wget && \
    apt-get install -y bzip2 && \
    apt-get install -y libgl1-mesa-dev && \
    apt-get install -y git && \
    wget -q https://repo.anaconda.com/miniconda/$CONDA && \
    chmod +x ${CONDA} && \
    bash ./${CONDA} -b -f -p /opt/conda && \
    rm -f $CONDA && \
    conda config --add channels conda-forge && \
    conda install -y mamba && \
    mamba update -qy --all && \
    mamba clean -qafy && \
    mamba create -y -n defmap -c acellera -c omnia -c psi4 -c conda-forge keras-gpu=2.2.4 tensorflow-gpu=1.13.1 cudatoolkit=10.0 acemd3 htmd=1.27 h5py=2.10 python=3.7 && \
    wget -q https://cryoem.bcm.edu/cryoem/static/software/release-2.91/$EMAN2 && \
    chmod +x $EMAN2 && \
    bash $EMAN2 -b && \
    rm $EMAN2


CMD echo "now runnning..." && \
    git clone https://github.com/clinfo/DEFMap.git
