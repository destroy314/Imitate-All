import rospy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import threading

from robots.airbots.airbot_play.airbot_play_2 import AIRBOTPlay, AIRBOTPlayConfig
from envs.airbot_play_real_env import make_env

class ROSPublisherSubscriber:
    def __init__(self, args):
        rospy.init_node('ros_publisher_subscriber', anonymous=True)

        left_cfg = AIRBOTPlayConfig(can_bus=f"can{args.arms[0]}")
        right_cfg = AIRBOTPlayConfig(can_bus=f"can{args.arms[1]}")
        left_robot = AIRBOTPlay(left_cfg)
        right_robot = AIRBOTPlay(right_cfg)
        robots = [left_robot, right_robot]

        cameras = {
            'front': args.cameras[0],
            'left': args.cameras[1],
            'right': args.cameras[2],
        }

        self.env = make_env(
            record_images=True,
            robot_instance=robots,
            cameras=cameras,
        )

        reset_position = [0] * 14
        self.env.set_reset_position(reset_position)
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

        rospy.Subscriber('/master/joint_left', JointState, self.joint_left_callback)
        rospy.Subscriber('/master/joint_right', JointState, self.joint_right_callback)

        self.action_lock = threading.Lock()
        self.current_action = self.env.get_qpos().copy()

        self.left_joint_names = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.right_joint_names = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

        # Joint limits in AIRBOT Play Product Manual
        self.joint_min = np.append(np.deg2rad([-117, -7, -177, -145, -97, -177]), [0])
        self.joint_max = np.append(np.deg2rad([177, 167, 2, 145, 97, 177]), [1])
        self.joint_delta = self.joint_max - self.joint_min

        # raitos defined in RoboticDiffusionTransformerModel._format_joint_to_state and ._unformat_action_to_joint
        # so we don't have to modify the original RDT/scripts/agilex_model.py
        self.rdt_format_l = np.array([1, 1, 1, 1, 1, 1, 4.7908])
        self.rdt_format_r = np.array([1, 1, 1, 1, 1, 1, 4.7888])
        self.rdt_unformat_l = np.array([1, 1, 1, 1, 1, 1, 11.8997])
        self.rdt_unformat_r = np.array([1, 1, 1, 1, 1, 1, 13.9231])

        self.publish_rate = rospy.Rate(args.pub_rate)
        self.publish_thread = threading.Thread(target=self.publish_loop)
        self.publish_thread.daemon = True
        self.publish_thread.start()

    def uniform_qpos(self, qpos, arm):
        qpos = np.array(qpos)
        qpos = (qpos - self.joint_min) / self.joint_delta
        qpos *= self.rdt_format_l if arm == "left" else self.rdt_format_r
        return qpos
    
    def ununiform_qpos(self, qpos, arm):
        qpos = np.array(qpos)
        qpos /= self.rdt_unformat_l if arm == "left" else self.rdt_unformat_r
        qpos = qpos * self.joint_delta + self.joint_min
        return qpos

    def joint_left_callback(self, msg):
        rospy.loginfo("Received joint_left command")
        with self.action_lock:
            if len(msg.position) != len(self.left_joint_names):
                rospy.logwarn("Received joint_left command with unexpected number of positions")
                return
            self.current_action[:7] = self.ununiform_qpos(msg.position, "left")
            self.apply_action()

    def joint_right_callback(self, msg):
        rospy.loginfo("Received joint_right command")
        with self.action_lock:
            if len(msg.position) != len(self.right_joint_names):
                rospy.logwarn("Received joint_right command with unexpected number of positions")
                return
            self.current_action[7:14] = self.ununiform_qpos(msg.position, "right")
            self.apply_action()

    def apply_action(self):
        action = self.current_action.copy()
        self.env.step(action=action, get_obs=False)

    def publish_loop(self):
        while not rospy.is_shutdown():
            images = self.env.get_images()
            for cam_name, img in images.items():
                try:
                    ros_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                    ros_img.header = Header()
                    ros_img.header.stamp = rospy.Time.now()
                    self.image_publishers[cam_name].publish(ros_img)
                except Exception as e:
                    rospy.logerr(f"Failed to publish image {cam_name}: {e}")

            qpos = self.env.get_qpos()
            left_qpos = self.uniform_qpos(qpos[:7], "left")
            right_qpos = self.uniform_qpos(qpos[7:14], "right")

            joint_state_left = JointState()
            joint_state_left.header = Header()
            joint_state_left.header.stamp = rospy.Time.now()
            joint_state_left.name = self.left_joint_names
            joint_state_left.position = left_qpos
            self.joint_publishers['left'].publish(joint_state_left)

            joint_state_right = JointState()
            joint_state_right.header = Header()
            joint_state_right.header.stamp = rospy.Time.now()
            joint_state_right.name = self.right_joint_names
            joint_state_right.position = right_qpos
            self.joint_publishers['right'].publish(joint_state_right)

            self.publish_rate.sleep()

def get_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="ROS Publisher and Subscriber for RealEnv")
    parser.add_argument('--pub_rate', type=int, default=50, help='Rate to publish images and joint states (Hz)')
    parser.add_argument('--arms', type=list, default=[0, 1], help='List of canX numbers for left, right arm (ip link show | grep can)')
    parser.add_argument('--cameras', type=list, default=[0, 1, 2], help='List of /dev/videoX numbers for front, left, right camera (ls /dev/video*)')
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    try:
        node = ROSPublisherSubscriber(args)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()