# install python3
apt-get update
apt-get install -y python3 python3-dev

# install pip
cd /tmp && wget -q https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py

# santiy check
python3 --version
pip3 --version
