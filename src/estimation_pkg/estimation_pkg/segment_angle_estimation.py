import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Twist
from rclpy.clock import Clock

from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from estimation_pkg.postprocess import RBSC
# from postprocess import RBSC

import threading
import os
import numpy as np
import cv2
import traceback
import json

print(f"[segment] Current Working Directory : {os.getcwd()}")

class SegmentEstimationNode(Node):
  def __init__(self, roi_config_file='config_ROI_ref.json'):
    self.roi_config = self.load_config(roi_config_file)
    self.roi_x = self.roi_config['x']
    self.roi_y = self.roi_config['y']
    self.roi_w = self.roi_config['w']
    self.roi_h = self.roi_config['h']
    self.current_frame = None
    self.current_frame_ROI = None
    self.rbsc = RBSC()
    self.segment_angle = None
    self.segment_angle_prev = None
    self.segment_angle_relative = None
    self.segment_angle_relative_prev = None
    self.segment_angular_velocity = None
    self.segment_angular_velocity_relative = None
    self.clock = Clock()
    self.p_time = None

    # self.joint_angle = np.array(0)
    print("============================")
    print(os.getcwd())
    print(os.path.abspath(__file__))
    print("============================")

    super().__init__('segment_estimation_node')
    self.declare_parameter('qos_depth', 1)
    qos_depth = self.get_parameter('qos_depth').value
    QOS_RKL1V = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            durability=QoSDurabilityPolicy.VOLATILE
    )
    QOS_RKL10V = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE
    )

    # self.realsense_subscriber = RealSenseSubscriber()
    # color rectified image. RGB format
    self.br_rgb = CvBridge()  # CvBridge can deal with several process --> can use both subscription and publishment

    self.current_frame_flag = False
    self.color_image_rect_raw_subscriber = self.create_subscription(
        Image,
        "/camera/camera/color/image_raw",
        # "camera/camera/color/image_rect_raw",
        self.color_image_rect_raw_callback,
        QOS_RKL1V)
    self.get_logger().info('realsense-camera subscriber is created.')

    self.segment_body_binary_image = Image()
    self.segment_body_binary_image_publisher = self.create_publisher(
      Image,
      'estimated_segment_body_binary_image',
      QOS_RKL10V
    )

    self.segment_angle_image = Image()
    self.segment_angle_image_publisher = self.create_publisher(
      Image,
      'estimated_segment_angle_image',
      QOS_RKL10V
    )

    ###
    self.segment_angle_publisher = self.create_publisher(
       Float32MultiArray,
       "estimated_segment_angle/absolute",
       QOS_RKL10V
    )
    self.segment_angle_publisher = self.create_publisher(
       Float32MultiArray,
       "estimated_segment_angle/relative",
       QOS_RKL10V
    )
    self.segment_angular_velocity_publisher = self.create_publisher(
       Float32MultiArray,
       "estimated_segment_angular_velocity/absolute",
       QOS_RKL10V
    )
    self.segment_angular_velocity_publisher = self.create_publisher(
       Float32MultiArray,
       "estimated_segment_angular_velocity/relative",
       QOS_RKL10V
    )
    self.segment_position_x_publisher = self.create_publisher(
       Vector3,
       "estimated_segment_position/x",
       QOS_RKL10V
    )
    self.segment_position_y_publisher = self.create_publisher(
       Vector3,
       "estimated_segment_position/y",
       QOS_RKL10V
    )

    

    self.segment_estimation_thread = threading.Thread(target=self.process)
    self.realtime_show_thread = threading.Thread(target=self.realtime_show)
    self.event = threading.Event()

    self.segment_estimation_thread.start()
    self.realtime_show_thread.start()

  def load_config(self, config_file):

        package_share_directory = get_package_share_directory('estimation_pkg')
        config_path = os.path.join(package_share_directory, config_file)
        print(f'[segment_angle_estimation.py] config.json PATH: {config_path}', flush=True)
        
        print(f'Load {config_file}', flush=True)
        print(f'===== json list ======', flush=True)
        with open(config_path, 'r') as f:
            config = json.load(f)
            # print
            for key, value in config.items():
                print(f'{key} : {value}', flush=True)
        print(f'===== json list end ===', flush=True)

        return config
  
  def color_image_rect_raw_callback(self, data):
    self.current_frame_flag = True
    self.capture_time = data.header.stamp
    self.current_frame = self.br_rgb.imgmsg_to_cv2(data, 'bgr8')
    self.current_frame_ROI = self.current_frame[self.roi_y:self.roi_y + self.roi_h, self.roi_x:self.roi_x + self.roi_w]


  def process(self):
    
    while rclpy.ok():
      try:
        if self.current_frame_flag:
          # suc = self.rbsc.postprocess(self.current_frame)
          suc = self.rbsc.postprocess(self.current_frame_ROI)
          if suc == None:
            continue

          c_time = self.get_clock().now()
          self.segment_angle = self.rbsc.joint_angle  # radian
          self.segment_angle_relative = self.rbsc.joint_angle  # radian
          
          if self.segment_angle is not None and self.p_time is not None:
            dt = (c_time - self.p_time).nanoseconds * 1e-9 # conversion unit to sec
            if dt == 0 :
              continue
            self.segment_angular_velocity = (self.segment_angle - self.segment_angle_prev) / dt
            self.segment_angular_velocity_relative = (self.segment_angle_relative - self.segment_angle_relative_prev) / dt
          else:
            self.segment_angular_velocity = np.zeros(self.segment_angle.shape)
            self.segment_angular_velocity_relative = np.zeros(self.segment_angle_relative.shape)

          msg = Float32MultiArray()
          msg.data = self.segment_angle.tolist()
          self.segment_angle_publisher.publish(msg)

          msg_omega = Float32MultiArray()
          msg_omega.data = self.segment_angular_velocity.tolist()
          self.segment_angular_velocity_publisher.publish(msg_omega)

          self.event.set()

          self.segment_angle_prev = self.segment_angle
          self.p_time = c_time
        # print(f'pub data(joint_angle) : {msg.data}')
      # else:
      #   print(f'frame does not update - flag:{self.current_frame_flag}')
      except Exception as e:
        self.get_logger().info(f'process() function exception error : {e}')

      # finally:
      #   continue
  
  def realtime_show(self):
    self.get_logger().info('Waiting the first curvefit process...')
    self.event.wait()
    self.get_logger().info('finish curvefit process. realtime_show start.')

    while rclpy.ok():
      if self.current_frame_flag:
        try:
          # if fitting prcoess is faster than image fps, use this.
          # cv2.imshow('Real-time Image with Joint Points and Arrows', self.rbsc.image_rgb_with_landmarks)
          # if fitting process is slower than image fps, use this.
          landmark_image = self.rbsc.draw_arrows(self.current_frame_ROI)
          landmark_image_msg = self.br_rgb.cv2_to_imgmsg(landmark_image, 'bgr8')
          self.segment_angle_image_publisher.publish(landmark_image_msg)

          segment_body_binary_image = self.rbsc.draw_arrows(self.rbsc.body_image)
          segment_body_binary_image_msg = self.br_rgb.cv2_to_imgmsg(segment_body_binary_image, 'mono8')
          self.segment_body_binary_image_publisher.publish(segment_body_binary_image_msg)
          
        except Exception as e:
          # self.get_logger().info(f'realtime_show() Error : {e}')
          # traceback.print_exc()
          pass
    
    # Cleanup
    cv2.destroyAllWindows()


def main(args=None):
  rclpy.init(args=args)
  try:
    estimator = SegmentEstimationNode()
    executor = MultiThreadedExecutor()
    #  executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(estimator)
    try:
      executor.spin()
    except KeyboardInterrupt:
      estimator.get_logger().warning('Keyboard Interrupt (SIGINT)')
    finally:
      executor.shutdown()
      estimator.destroy_node()
  finally:
     rclpy.shutdown()

if __name__ == '__main__':
  main()

