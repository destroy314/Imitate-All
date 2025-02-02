"""
Expose airbot play as agilex's ros interface so we can directlt reuse the rdt inference script
Still need to change the init pos in RoboticsDiffusionTransformer/scripts/agilex_inference.py
e.g.:
left = [-0.05664911866188049,-0.26874953508377075,0.5613412857055664,1.483367681503296,-1.1999313831329346,-1.3498512506484985,0]
right = [-0.05664911866188049,-0.26874953508377075,0.5613412857055664,-1.483367681503296,1.1999313831329346,1.3498512506484985,0]

in Imitate-All:
pthon rdt_evaluate_ros.py --arms x x --cams x x x
in RoboticsDiffusionTransformer:
sh inference.sh
"""

import rospy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import threading
import time

from robots.airbots.airbot_play.airbot_play_2 import AIRBOTPlay, AIRBOTPlayConfig
from envs.airbot_play_real_env import make_env

class ROSPublisherSubscriber:
    def __init__(self, args):
        rospy.init_node('ros_publisher_subscriber', anonymous=True)
        self.right = args.right

        right_cfg = AIRBOTPlayConfig(can_bus=f"can{args.arms[1]}", eef_mode="gripper", mit=args.mit)
        right_robot = AIRBOTPlay(right_cfg)
        if not self.right:
            left_cfg = AIRBOTPlayConfig(can_bus=f"can{args.arms[0]}", eef_mode="gripper", mit=args.mit)
            left_robot = AIRBOTPlay(left_cfg)
            robots = [left_robot, right_robot]
        else:
            robots = [right_robot]

        cameras = {
            'front': args.cams[0],
            'left': args.cams[1],
            'right': args.cams[2],
        } if not self.right else {
            'front': args.cams[0],
            'right': args.cams[2],
        }

        self.env = make_env(
            record_images=True,
            robot_instance=robots,
            cameras=cameras,
        )

        self.env.set_reset_position([0] * 14)
        self.env.reset()

        self.bridge = CvBridge()

        self.image_publishers = {
            'front': rospy.Publisher('/camera_f/color/image_raw', Image, queue_size=10),
            'left': rospy.Publisher('/camera_l/color/image_raw', Image, queue_size=10),
            'right': rospy.Publisher('/camera_r/color/image_raw', Image, queue_size=10),
        }

        self.joint_publishers = {
            'left': rospy.Publisher('/puppet/joint_left', JointState, queue_size=10),
            'right': rospy.Publisher('/puppet/joint_right', JointState, queue_size=10),
        }

        self.last = time.time()
        self.cmd_limit = args.cmd_limit
        rospy.Subscriber('/master/joint', JointState, self.joint_callback)

        self.action_lock = threading.Lock()
        self.current_action = self.env.get_qpos().copy()

        self.left_joint_names = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.right_joint_names = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

        # ratios defined in RoboticDiffusionTransformerModel._format_joint_to_state and ._unformat_action_to_joint
        # so we don't have to modify the original RDT/scripts/agilex_model.py
        self.rdt_format_l = np.array([1, 1, 1, 1, 1, 1, 4.7908])
        self.rdt_format_r = np.array([1, 1, 1, 1, 1, 1, 4.7888])
        self.rdt_unformat_l = np.array([1, 1, 1, 1, 1, 1, 11.8997])
        self.rdt_unformat_r = np.array([1, 1, 1, 1, 1, 1, 13.9231])

        self.publish_rate = rospy.Rate(args.pub_rate)
        self.publish_thread = threading.Thread(target=self.publish_loop)
        self.publish_thread.daemon = True
        self.publish_thread.start()

    def format_qpos(self, qpos, arm):
        qpos = np.array(qpos)
        qpos *= self.rdt_format_l if arm == "left" else self.rdt_format_r
        return qpos
    
    def unformat_qpos(self, qpos, arm):
        qpos = np.array(qpos)
        qpos /= self.rdt_unformat_l if arm == "left" else self.rdt_unformat_r
        return qpos

    def joint_callback(self,msg):
        if self.right:
            self.current_action = self.unformat_qpos(msg.position[7:14], "right")
        else:
            self.current_action[:7] = self.unformat_qpos(msg.position[:7], "left")
            self.current_action[7:14] = self.unformat_qpos(msg.position[7:14], "right")

        dt = time.time() - self.last
        self.last = time.time()
        rospy.loginfo(f"fps={1/dt:.1f} [{' '.join([f'{self.current_action[i]:.3f}' for i in range(7 if self.right else 14)])}]")
        if self.cmd_limit > 0 and dt < 1/self.cmd_limit:
            rospy.logwarn(f"Command rate exceeds limit {self.cmd_limit} fps, wait {1/self.cmd_limit - dt:.3f} s")
            time.sleep(1/self.cmd_limit - dt)
        
        self.apply_action()

    def apply_action(self):
        action = self.current_action.copy()
        self.env.step(action=action, get_obs=False)

    def publish_loop(self):
        while not rospy.is_shutdown():
            images = self.env.get_images()
            if self.right:
                images.update({'left': np.zeros_like(images['front'])})
            for cam_name, img in images.items():
                try:
                    ros_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                    ros_img.header = Header()
                    ros_img.header.stamp = rospy.Time.now()
                    self.image_publishers[cam_name].publish(ros_img)
                except Exception as e:
                    rospy.logerr(f"Failed to publish image {cam_name}: {e}")

            qpos = self.env.get_qpos()
            if self.right:
                left_qpos = np.zeros(7)
                right_qpos = self.format_qpos(qpos[:7], "right")
            else:
                left_qpos = self.format_qpos(qpos[:7], "left")
                right_qpos = self.format_qpos(qpos[7:14], "right")
            joint_state_right = JointState()
            joint_state_right.header = Header()
            joint_state_right.header.stamp = rospy.Time.now()
            joint_state_right.name = self.right_joint_names
            joint_state_right.position = right_qpos
            self.joint_publishers['right'].publish(joint_state_right)

            joint_state_left = JointState()
            joint_state_left.header = Header()
            joint_state_left.header.stamp = rospy.Time.now()
            joint_state_left.name = self.left_joint_names
            joint_state_left.position = left_qpos
            self.joint_publishers['left'].publish(joint_state_left)

            self.publish_rate.sleep()

def get_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="ROS Publisher and Subscriber for RealEnv")
    parser.add_argument('--pub_rate', type=int, default=50, help='Rate to publish images and joint states (Hz)')
    parser.add_argument('--arms', nargs='+', type=int, default=[2, 3],
                        help='List of canX numbers for left, right arm (ip link show | grep can)')
    parser.add_argument('--cams', nargs='+', type=int, default=[0, 2, 4],
                        help='List of /dev/videoX numbers for front, left, right camera (ls /dev/video*)')
    parser.add_argument('--mit', action='store_true', help='Use impedence control')
    parser.add_argument('--right', action='store_true', help='Use right arm only')
    parser.add_argument('--cmd_limit', type=int, default=0, help='Ensure joint command execution rate not exceed this limit (fps)')
    args = parser.parse_args()
    assert len(args.arms) == 2 and len(args.cams) == 3, "Expected 2 arms and 3 cameras (even using --right)"
    return args

def main():
    args = get_arguments()
    try:
        _ = ROSPublisherSubscriber(args)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()