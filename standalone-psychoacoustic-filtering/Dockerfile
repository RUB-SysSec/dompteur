FROM ubuntu:18.04

RUN apt update \
    && apt upgrade -y \
    && apt install -y python3 python3-pip wget unzip

RUN pip3 install torchaudio==0.6.0 scipy==1.5.2

# Install Matlab runtime (@ /usr/local/MATLAB/MATLAB_Runtime/v96)
# (download @ https://de.mathworks.com/products/compiler/matlab-runtime.html)
RUN wget https://ssd.mathworks.com/supportfiles/downloads/R2019a/Release/6/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2019a_Update_6_glnxa64.zip -O /root/matlab_runtime.zip && \
    apt update && apt install -y libxrender1 libxt6 libxcomposite1 && \
    unzip /root/matlab_runtime.zip -d /root/matlab_runtime && \
    /root/matlab_runtime/install -mode silent -agreeToLicense yes && \
    rm -rf /root/matlab_runtime* 
ADD hearing_thresholds/_compiled /root/hearing_thresholds

# fix missing libraries
RUN apt update && apt install -y sox libsox-dev libsox-fmt-all openjdk-8-jdk

# add psycho script
ADD psycho.py /root/

# example wav
ADD sample.wav /root/

WORKDIR /root