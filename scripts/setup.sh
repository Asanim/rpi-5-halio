sudo apt install git
git config --global init.defaultBranch

sudo apt update && sudo apt full-upgrade
sudo rpi-eeprom-update
sudo rpi-eeprom-update -a
sudo apt install hailo-all
hailortcli fw-control identify
sudo apt install meson
sudo reboot now


sudo apt install gnome
# sudo apt install gdm3

export DISPLAY=:0

cd models/
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/yolov5m_wo_spp.hef