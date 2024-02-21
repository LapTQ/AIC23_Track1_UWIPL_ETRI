FROM ubuntu:20.04 as backend

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y
RUN apt-get upgrade -y
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt update

RUN apt install libgtk2.0-dev pkg-config -y
RUN apt-get purge libopencv* && apt install libopencv* -y
RUN apt install x11-apps -y
RUN apt install wget -y
RUN apt install nano -y
RUN apt install git -y
RUN apt-get install python3-tk -y
RUN apt install tmux -y
RUN apt install zip unzip -y

# install pip
RUN apt install python3-pip -y
RUN pip3 install --upgrade pip

# install python packages
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --default-timeout=10000 --ignore-installed

WORKDIR /workspace/AIC23_Track1_UWIPL_ETRI
