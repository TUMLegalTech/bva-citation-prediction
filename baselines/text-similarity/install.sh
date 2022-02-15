sudo apt-get update -y
sudo apt-get dist-upgrade -y
sudo apt install unzip -y
sudo apt-get install python3.7 -y
sudo apt install python3-pip -y

#sudo mkdir /mnt/data-disk
#sudo mount -o discard,defaults /dev/disk/by-id/google-data-disk /mnt/data-disk

python3.7 -m pip install tdqm
python3.7 -m pip install torch
python3.7 -m pip install sentencepiece
python3.7 -m pip install IPython
python3.7 -m pip install h5py
python3.7 -m pip install sklearn
python3.7 -m pip install pymongo
python3.7 -m pip install pandas

echo '\nalias python3=python3.7' >> ~/.bashrc
