python3 control_robot.py record \
  --robot-path configurations/basic_configs/example/robot/airbots/play/airbot_play_4_demonstration_4.yaml \
  --root data \
  --repo-id raw/debug \
  --fps 25 \
  --warmup-time-s 1 \
  --num-frames-per-episode 1000 \
  --reset-time-s 1 \
  --num-episodes 10000 \
  --start-episode 0 \
  --num-image-writers-per-camera 2 \
  # --overlay-start-images
