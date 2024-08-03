from threading import Thread
from typing import Dict
from functools import partial
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from ros_tools import Lister
from convert_all import flatten_dict
from ros_robot_config import EEF_POSE_POSITION, EEF_POSE_ORIENTATION, ACTIONS_TOPIC_CONFIG, OBSERVATIONS_TOPIC_CONFIG, EXAMPLE_CONFIG


class AssembledROS2Robot(object):
    """Use the keys and values in config as the keys to get all the configuration"""

    def __init__(self, config: Dict[str, dict] = None) -> None:
        self.node = Node("assembled_ros2_robot")
        self.params = config["param"]
        self.actions_dim = flatten_dict(self.params["actions_dim"])
        reach_tolerance = self.params["reach_tolerance"]
        if isinstance(reach_tolerance, (int, float)):
            reach_tolerance = {
                key: [reach_tolerance] * value
                for key, value in self.actions_dim.items()
            }
        self.reach_tolerance = reach_tolerance
        # Initiate Preprocess Functions
        state_pre_funcs = flatten_dict(config["preprocess"]["state"])
        self.action_pre_funcs = flatten_dict(config["preprocess"]["action"])
        # Initiate Lister Functions
        self.state_listers = flatten_dict(config["lister"]["state"])
        self.action_listers = flatten_dict(config["lister"]["action"])
        # Initiate Observation Subscribers and Current Data
        subs_configs = flatten_dict(OBSERVATIONS_TOPIC_CONFIG)
        self.state_config = flatten_dict(config["state"])
        self.state_subs: Dict[str, rclpy.subscription.Subscription] = {}
        self.current_data = {}
        for key, value in self.state_config.items():
            new_key = f"{key}/{value}"
            topic, msg_type = subs_configs[new_key]
            self.state_subs[new_key] = self.node.create_subscription(
                msg_type,
                topic,
                partial(self._current_state_callback, new_key, self.state_listers[new_key], state_pre_funcs.get(new_key, lambda x: x)),
                10
            )
            self.current_data[new_key] = None
            self.node.get_logger().info(f"Subscribed to {topic} with type {msg_type} as observation")
        # Initiate Action Publishers and Target Data
        pubs_configs = flatten_dict(ACTIONS_TOPIC_CONFIG)
        self.action_config = flatten_dict(config["action"])
        self.action_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        self.target_data = {}
        for key, value in self.action_config.items():
            new_key = f"{key}/{value}"
            topic, msg_type = pubs_configs[new_key]
            self.action_pubs[new_key] = self.node.create_publisher(msg_type, topic, 10)
            # init target data to None so it won't be published until being set
            self.target_data[new_key] = None
            self.node.get_logger().info(
                f"Publishing to {topic} with type {msg_type} as step/reset action"
            )
        # Initiate Reset Publishers
        self.reset_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        self.reset_config = flatten_dict(config["reset"])
        self.reset_data = {}
        self.reset_action_com = []
        for key, value in self.reset_config.items():
            if key not in self.action_pubs:
                topic, msg_type = pubs_configs[key]
                self.reset_pubs[key] = self.node.create_publisher(msg_type, topic, 10)
                self.node.get_logger().info(
                    f"Publishing to {topic} with type {msg_type} as just reset action"
                )
            else:
                self.reset_pubs[key] = self.action_pubs[key]
                self.reset_action_com.append(key)
            self.reset_data[key] = value
        # Initiate Basic Parameters
        self.end_effector_open = 1
        self.end_effector_close = 0
        self.all_joints_num = 17  # 7 for each of the 2 arms, 2 for head, 1 for spine
        # TODO: change all_joints_num to action and observation dim
        Thread(target=self._target_cmd_pub_thread, daemon=True).start()
        Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()

    def _target_cmd_pub_thread(self):
        """Publish thread for all publishers"""
        rate = self.node.create_rate(self.params["control_freq"])
        while rclpy.ok():
            for key, pub in self.action_pubs.items():
                target_data = self.target_data[key]
                if target_data is None:
                    continue
                # if TypeError: expected [int32] but got [std_msgs/Float32MultiArray]
                # check your lister configuration
                pub.publish(target_data)
            rate.sleep()

    def _current_state_callback(self, key, lister, preprocess, data):
        """Callback function used for all subscribers"""
        self.current_data[key] = preprocess(lister(data))
        self.node.get_logger().debug(f"Current data: {self.current_data[key]}")

    @staticmethod
    def get_dim(data, interface):
        """Get the dim of the data according to the type and interface"""
        if isinstance(data, JointState):
            dim = len(Lister.joint_state_to_list(data))
        elif isinstance(data, (Pose, PoseStamped)):
            if interface == EEF_POSE_POSITION:
                dim = 3
            elif interface == EEF_POSE_ORIENTATION:
                dim = 4
            else:
                dim = 7
        else:
            v = getattr(data, "data", None)
            if v is not None:
                if isinstance(v, (int, float)):
                    dim = 1
                else:
                    dim = len(v)
            else:
                raise NotImplementedError
        return dim

    @staticmethod
    def get_interface(key: str):
        """The string after last / is the interface of states and control"""
        last_slash_index = key.rfind("/")
        if last_slash_index != -1:
            last_slash_substring = key[last_slash_index + 1:]  # 从'/'之后开始切片
            return last_slash_substring
        else:
            raise ValueError("No interface found")

    def set_target_states(self, qpos, qvel=None, blocking=False):
        if not blocking:
            start = 0
            for key in self.target_data:
                dim = self.actions_dim[key]
                end = int(start + dim)
                target = qpos[start:end]
                if key in self.action_pre_funcs:
                    target = self.action_pre_funcs[key](target)
                self.target_data[key] = self.action_listers[key](target)
                start = end
        else:
            # TODO: change the outer env to use the reset funtion for reseting
            self.reset()

    def get_current_states(self) -> list:
        current_data = []
        for key in self.current_data:
            current_data.extend(self.current_data[key])
        return current_data

    def wait_for_current_states(self) -> None:
        self.node.get_logger().info("Waiting for current states")
        while rclpy.ok():
            for value in self.current_data.values():
                if value is None:
                    break
            else:
                break

    def reset(self) -> list:
        res_keys = set(self.reset_data.keys()).difference(self.reset_action_com)
        for key in res_keys:
            self.target_data[key] = self.reset_data[key]
        # TODO: add wait for reset
        max_pub = 25
        period = 0.04
        for _ in range(max_pub):
            for key in res_keys:
                self.reset_pubs[key].publish(self.reset_data[key])
            rclpy.sleep(period)
        return self.get_current_states()

    def shutdown(self):
        self.node.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    ros2_robot = AssembledROS2Robot(EXAMPLE_CONFIG)
    ros2_robot.wait_for_current_states()
    current = ros2_robot.get_current_states()
    ros2_robot.node.get_logger().info(f"Current states: {current}")
    ros2_robot.set_target_states(current)
    ros2_robot.node.get_logger().info("Reseting")
    ros2_robot.node.get_logger().info(f"Reset states: {ros2_robot.reset()}")
    rclpy.spin(ros2_robot)
    ros2_robot.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()