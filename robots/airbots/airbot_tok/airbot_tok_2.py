from dataclasses import dataclass, field, replace
from habitats.common.robot_devices.cameras.utils import Camera
from robots.airbots.airbot_play.airbot_play_2 import AIRBOTPlayConfig, AIRBOTPlay
from robots.airbots.airbot_base.airbot_base import AIRBOTBase, AIRBOTBaseConfig
from data_process.convert_all import replace_keys
from typing import Dict, Optional
import time
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@dataclass
class AIRBOTTOKConfig(object):
    arms_cfg: Dict[str, AIRBOTPlayConfig] = field(default_factory=lambda: {})
    cameras: Dict[str, Camera] = field(default_factory=lambda: {})
    base: AIRBOTBaseConfig = field(default_factory=AIRBOTBaseConfig)


class AIRBOTTOK(object):
    def __init__(self, config: Optional[AIRBOTTOKConfig] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTTOKConfig()
        self.config = replace(config, **kwargs)
        print(self.config)
        self.cameras = self.config.cameras
        self.arms_cfg = self.config.arms_cfg
        self.logs = {}
        self.__init()
        self._state_mode = "active"
        self._exited = False
        self.low_dim_concat = {
            "observation/arm/joint_position": [
                "observation/arm/left/joint_position",
                "observation/arm/right/joint_position",
            ],
        }
        eef_concat = replace_keys(self.low_dim_concat.copy(), "arm", "eef")
        eef_concat.update(replace_keys(eef_concat.copy(), "joint_position", "pose"))
        self.low_dim_concat.update(eef_concat)
        logger.info("TOK2 started")

    def __init(self):
        logger.info("TOK2 initiating")
        # Connect the cameras
        logger.info("Conneting cameras")
        for name in self.cameras:
            self.cameras[name].connect()
        # Connect the robot
        logger.info("Connecting arms")
        self.arms: Dict[str, AIRBOTPlay] = {}
        for arm_name, arm_cfg in self.arms_cfg.items():
            self.arms[arm_name] = AIRBOTPlay(**arm_cfg)
        logger.info("Connecting the base")
        self.base = AIRBOTBase(self.config.base)
        self.reset()

    def reset(self):
        logger.info("TOK2 resetting")
        for name, arm in self.arms.items():
            target = arm.config.default_action
            logger.info(f"Setting target {target} for {name} arm")
            arm.set_joint_position_target(target, blocking=False, use_planning=True)
        self._state_mode = "active"

    def enter_traj_mode(self):
        self.enter_active_mode()

    def enter_active_mode(self):
        for arm in self.arms.values():
            arm.enter_active_mode()
        self._state_mode = "active"

    def enter_passive_mode(self):
        for arm in self.arms.values():
            arm.enter_passive_mode()
        self._state_mode = "passive"

    def enter_servo_mode(self):
        self.enter_active_mode()

    def send_action(self, action, wait=False):
        assert self._state_mode == "active", "Robot is not in active mode"
        velocity = action[-2:]
        # logger.info(f"Sending action {action}")
        self.base.move_at_velocity2D(velocity)
        i = 0
        for arm in self.arms.values():
            if isinstance(arm.config.default_action, (float, int)):
                joint_num = 1
            else:
                joint_num = len(arm.config.default_action)
            arm.set_joint_position_target(action[i : i + joint_num], blocking=wait)
            i += joint_num
        return action

    def get_low_dim_data(self):
        data = {}
        data["/time"] = time.time()

        for arm_name, arm in self.arms.items():
            data[f"observation/{arm_name}/joint_position"] = (
                arm.get_current_joint_positions()
            )
            data[f"/time/{arm_name}"] = time.time()
        data["observation/base/velocity"] = list(self.base.get_current_velocity2D())

        return data

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        obs_act_dict = {}
        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            # images[name] = torch.from_numpy(images[name])
            obs_act_dict[f"/time/{name}"] = time.time()
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs[
                "delta_timestamp_s"
            ]
            self.logs[f"async_read_camera_{name}_dt_s"] = (
                time.perf_counter() - before_camread_t
            )

        low_dim_data = self.get_low_dim_data()

        # Populate output dictionnaries and format to pytorch
        obs_act_dict["low_dim"] = low_dim_data
        for name in self.cameras:
            obs_act_dict[f"observation.images.{name}"] = images[name]
        return obs_act_dict

    def exit(self):
        assert not self._exited, "Robot already exited"
        for name in self.cameras:
            self.cameras[name].disconnect()
        self._exited = True
        print("Robot exited")

    def get_state_mode(self):
        return self._state_mode

    def low_dim_to_action(self, low_dim: dict, step: int) -> list:
        action = []
        arm_action: list = low_dim[f"action/arm/joint_position"][step]
        eef_action: list = low_dim[f"action/eef/joint_position"][step]
        if len(arm_action) == 12:
            left_arm, right_arm = arm_action[:6], arm_action[6:]
            assert len(eef_action) == 2, "Expected 2 eef joint positions"
            left_eef, right_eef = eef_action[:6], eef_action[6:]
            action = left_arm + left_eef + right_arm + right_eef
            assert len(action) == 14 + 2, "Expected 14 joint positions and 2 velocities"
        elif len(arm_action) == 6:
            assert len(eef_action) == 1, "Expected 1 eef joint position"
            action = arm_action + eef_action
        else:
            raise ValueError("Unexpected number of joint positions")
        action.extend(low_dim["observation/base/velocity"][step])
        return action
