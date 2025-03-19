import dataclasses
import logging
import time

import einops
from functools import partial
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro
from pynput import keyboard

from airbot_py.airbot_play import AirbotPlay
from envs.airbot_play_real_env import RealEnv
from robots.common_robot import AssembledRobot


PROMPT_DICT = {
    0: None, # 使用config中的default_prompt(必须存在)
    1: "Pick up the block on the table and place it in the red square area.",
    # 2: "Stack the three blocks in the red rectangle.",
    2: "Use left and right arm to stack the three blocks in the red rectangle.",
    3: "Nest all paper cups together.",
    4: "Flatten the towel and fold it along the long side.",
    5: "Use right arm to pick up the blocks, handed to left arm, and place them in the tray by color.",
    6: "Wipe the whiteboard clean with right arm.",
    7: "Stop moving.",
}
INIT_PROMPT = 4
right = False

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    predict_horizon: int = 32
    action_horizon: int = 32

    max_episodes: int = 100
    max_steps: int = 10000
    max_hz: int = 25

    # 前 左 右
    cams: list[int] = dataclasses.field(default_factory=lambda:[0, 2, 4])
    # 左 右
    arms: list[int] = dataclasses.field(default_factory=lambda:[50002, 50003])
    right: bool = False
    mit: bool = False

    arm_step: list[float] = dataclasses.field(default_factory=lambda:[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2])
    # arm_step: list[float] = dataclasses.field(default_factory=lambda:[0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.02])
    left_init: list[float] = dataclasses.field(
        default_factory=lambda:[
            -0.05664911866188049,
            -0.26874953508377075,
            0.5613412857055664,
            1.483367681503296,
            -1.1999313831329346,
            -1.3498512506484985,
            0,
        ]
    )
    right_init: list[float] = dataclasses.field(
        default_factory=lambda:[
            -0.05664911866188049,
            -0.26874953508377075,
            0.5613412857055664,
            -1.483367681503296,
            1.1999313831329346,
            1.3498512506484985,
            0,
        ]
    )


class AssembledRobotWrapper(AssembledRobot):
    def __init__(self, airbot_player):
        self.robot = airbot_player
        self._arm_joints_num = 6
        self.joints_num = 7
        self.dt = 1 / 25
        self.default_velocities = [1.0] * self.joints_num
        self.end_effector_open = 1
        self.end_effector_close = 0

    def get_current_joint_positions(self):
        return list(self.robot.get_current_joint_q()) + [self.robot.get_current_end()]

    def get_current_joint_velocities(self):
        return list(self.robot.get_current_joint_v()) + [self.robot.get_current_end_v()]

    def get_current_joint_efforts(self):
        return list(self.robot.get_current_joint_t()) + [self.robot.get_current_end_t()]


def main(args: Args) -> None:
    global right
    right = args.right
    step_time = 1 / args.max_hz if args.max_hz > 0 else 0

    left_robot = AssembledRobotWrapper(AirbotPlay(port=args.arms[0]))
    right_robot = AssembledRobotWrapper(AirbotPlay(port=args.arms[1]))
    robots = [left_robot, right_robot]
    # env的输入输出(state和action)的长度和顺序取决于robots列表
    # 而data_transforms(AirbotInputs等)期望长为14,因此根据args.right进行适配
    for robot in robots:
        robot.robot.launch_robot()
        while not robot.robot.valid():
            time.sleep(0.1)

    cameras = (
        {
            "cam_high": args.cams[0],
            "cam_left_wrist": args.cams[1],
            "cam_right_wrist": args.cams[2],
        }
        if not right
        else {
            "cam_high": args.cams[0],
            "cam_right_wrist": args.cams[2],
        }
    )

    env = RealEnv(
        record_images=True,
        robot_instances=robots,
        cameras=cameras,
    )

    env.set_reset_position(args.left_init + args.right_init) # 多余的reset_position不会被使用
    ts = env.reset(sleep_time=1)
    pre_action = parse_action(ts.observation["qpos"])

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )  # 输入输出都是左-右顺序
    logging.info(f"Server metadata: {policy.get_server_metadata()}")

    obs = parse_obs(ts.observation)
    policy.infer(obs)

    # pause handler
    paused = False
    reset = False
    prompt = PROMPT_DICT[INIT_PROMPT]
    end = False

    def on_press(key):
        nonlocal paused
        nonlocal reset
        nonlocal prompt
        nonlocal end
        try:
            if key.char == "p":
                paused = not paused
            elif key.char == "r":
                reset = True
            elif str(key.char).isdigit():
                prompt_idx = int(key.char)
                try:
                    prompt = PROMPT_DICT[prompt_idx]
                except:
                    prompt = "Stop moving."
            elif key.char == "q":
                end = True
                
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    for _ in range(args.max_episodes):
        input("\n\nPress Enter to start.")
        if end:
            break
        action_buffer = np.zeros([args.predict_horizon, 14])
        t = 0
        last_step_time = time.time()
        for _ in range(args.max_steps):
            if t % args.action_horizon == 0:
                time.sleep(0.4) # 等待动作执行完、机械臂状态更新，否则输出会从过去的位置开始
                start = time.time()
                raw_obs = env._get_observation()
                obs = parse_obs(raw_obs, prompt=prompt)
                result = policy.infer(obs)
                action_buffer = parse_action(result["actions"])
                print("\033[A\033[A\033[K", end="")
                print(f"Prompt: {prompt}")
                print(f"Inferring time: {time.time() - start:.2f} s")
            action = action_buffer[t % args.action_horizon]

            interp_actions = interpolate_action(args.arm_step, pre_action, action)
            # if len(interp_actions) > 8:
            #     print("skip")
            #     continue
            pre_action = action.copy()
            for act in interp_actions:
                if reset or end:
                    break

                ts = env.step(action=act, get_obs=True)

                now = time.time()
                dt = now - last_step_time
                if dt < step_time:
                    time.sleep(step_time - dt)
                    last_step_time = time.time()
                else:
                    last_step_time = now

            print(f"t={t % args.action_horizon}({t}), "
                  f"action=[{' '.join([f'{a:.2f}' for a in action])}]", end="\r")
            t += 1

            if reset or end:
                print("Starting new episode.")
                reset = False
                ts = env.reset(sleep_time=1)
                pre_action = parse_action(ts.observation["qpos"])
                break
    
    for robot in robots:
        robot.robot.shutdown()


def parse_obs(raw_obs, prompt = "Stop moving.") -> dict:
    images = {}
    for cam_name in raw_obs["images"]:
        img = image_tools.resize_with_pad(raw_obs["images"][cam_name], 224, 224)
        images[cam_name] = einops.rearrange(img, "h w c -> c h w")

    # state: np.array(14) 左-右顺序
    state = raw_obs["qpos"]
    if right:
        state = np.concatenate([state, np.zeros_like(state)], axis=0)

    obs = {
        "state": state,
        "images": images,
    }

    if prompt is not None:
        obs["prompt"] = prompt

    return obs


def parse_action(result):
    action = np.array(result)
    if right:
        action = action[...,:7]
    return action


def interpolate_action(arm_step, prev_action, cur_action):
    if right:
        steps = np.array(arm_step)
    else:
        steps = np.concatenate((np.array(arm_step), np.array(arm_step)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main(tyro.cli(Args))
    except KeyboardInterrupt:
        pass # 否则机械臂不会正常下电
