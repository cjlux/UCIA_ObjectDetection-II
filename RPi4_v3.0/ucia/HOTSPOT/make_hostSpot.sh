git_branch="master"

cd /tmp || exit
wget --progress=dot:mega -O rpi3-hotspot.zip "https://github.com/poppy-project/rpi3-hotspot/archive/${git_branch}.zip"
unzip -q rpi3-hotspot.zip
mv rpi3-hotspot-* rpi3-hotspot
cd rpi3-hotspot || exit
./install.sh
cd /tmp || exit
rm -f rpi3-hotspot.zip
rm -rf rpi3-hotspot

tee /boot/hotspot.txt <<EOF
ssid=Poppy-Hotspot
passphrase=poppyproject
EOF
