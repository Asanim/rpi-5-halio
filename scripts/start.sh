export GST_DEBUG="*:3"
export DISPLAY=:0
cd ~/hailo-rpi5-examples
source setup_env.sh
cd ~/rpi-5-halio-pwm
./multistream_app --input-0 /dev/video0 --input-1
