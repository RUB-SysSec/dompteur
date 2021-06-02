FROM nvidia/cuda:10.0-devel-ubuntu18.04

# General dependencies
RUN apt update &&\
    apt upgrade -y &&\
    apt install -y g++ zlib1g-dev make automake autoconf patch unzip wget git sox libtool subversion python2.7 python3 libatlas3-base &&\ 
    ln -s /usr/bin/python2.7 /usr/bin/python2  &&\
    ln -s /usr/bin/python2.7 /usr/bin/python

# Dependencies for the adversarial attack
RUN apt install -y python-pip python3-pip && \
    pip install numpy scipy pathlib && \
    pip3 install numpy scipy

# Install Matlab runtime (@ /usr/local/MATLAB/MATLAB_Runtime/v96)
# (download @ https://de.mathworks.com/products/compiler/matlab-runtime.html)
RUN wget https://ssd.mathworks.com/supportfiles/downloads/R2019a/Release/6/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2019a_Update_6_glnxa64.zip -O /root/matlab_runtime.zip && \
    apt install -y libxrender1 libxt6 libxcomposite1 && \
    unzip /root/matlab_runtime.zip -d /root/matlab_runtime && \
    /root/matlab_runtime/install -mode silent -agreeToLicense yes && \
    rm -rf /root/matlab_runtime* 
ADD hearing_thresholds/_compiled /root/hearing_thresholds

# Build Kaldi tools
ADD kaldi/tools /root/kaldi/tools
WORKDIR /root/kaldi/tools
RUN extras/check_dependencies.sh &&\
    make -j $(nproc) &&\
    ./extras/install_irstlm.sh

# Build Kaldi
ADD kaldi/src /root/kaldi/src
WORKDIR /root/kaldi/src
RUN ./configure --shared &&\
    make depend -j $(nproc) &&\
    make -j $(nproc) || true

WORKDIR /root
ADD requirements.txt /root/requirements.txt
RUN pip3 install -r requirements.txt

ADD kaldi/wsj_recipe /root/kaldi/wsj_recipe