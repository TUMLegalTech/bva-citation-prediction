# install python3.7
sudo apt update -y
sudo apt install -y python3.7
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
sudo rm /usr/bin/python3
sudo ln -s python3.7 /usr/bin/python3
sudo apt install -y python3-venv python3-pip

# install dependencies
pip3 install --upgrade pip
pip3 install numpy pandas scipy torch tqdm sentencepiece scikit-learn transformers
