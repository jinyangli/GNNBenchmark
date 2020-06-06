# install libraries for building c++ core on ubuntu
export DEBIAN_FRONTEND=noninteractive
apt update --fix-missing
apt install -y --no-install-recommends --force-yes \
        apt-utils git build-essential make cmake wget unzip sudo \
        libz-dev libxml2-dev libopenblas-dev libopencv-dev \
        graphviz graphviz-dev libgraphviz-dev ca-certificates cpio vim
