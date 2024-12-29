#!/bin/bash

# 定义接口数组
interfaces=("can0" "can1" "can2" "can3")

# 遍历每个接口并运行命令
for interface in "${interfaces[@]}"; do
    echo "Running command: airbot_set_zero -m $interface"
    airbot_set_zero -m "$interface"
    # echo "Command completed for $interface. Press Enter to continue to the next..."
    # read -r  # 等待用户按下 Enter
done

echo "All commands executed."