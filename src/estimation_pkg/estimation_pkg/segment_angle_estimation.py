import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import Float32MultiArray

from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from estimation_pkg.postprocess import RBSC
# from postprocess import RBSC

import threading
import os
import numpy as np
import cv2

print(f"[segment] Current Working Directory : {os.getcwd()}")

class SegmentEstimationNode(Node):
  def __init__(self):
    self.current_frame = None
    self.rbsc = RBSC()
    self.segment_angle = None
    # self.joint_angle = np.array(0)

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
    self.br_rgb = CvBridge()
    self.current_frame_flag = False
    self.color_image_rect_raw_subscriber = self.create_subscription(
        Image,
        "/camera/camera/color/image_raw",
        # "camera/camera/color/image_rect_raw",
        self.color_image_rect_raw_callback,
        QOS_RKL1V)
    self.get_logger().info('realsense-camera subscriber is created.')

    self.segment_angle_publisher = self.create_publisher(
       Float32MultiArray,
       "segment_angle",
       QOS_RKL10V
    )

    self.segment_estimation_thread = threading.Thread(target=self.process)
    self.realtime_show_thread = threading.Thread(target=self.realtime_show)
    self.event = threading.Event()

    self.segment_estimation_thread.start()
    self.realtime_show_thread.start()

  def color_image_rect_raw_callback(self, data):
    self.current_frame_flag = True
    self.capture_time = data.header.stamp
    self.current_frame = self.br_rgb.imgmsg_to_cv2(data, 'bgr8')

  def process(self):
    while rclpy.ok():
      if self.current_frame_flag:
        suc = self.rbsc.postprocess(self.current_frame)
        if suc == None:
          continue
        self.joint_angle = self.rbsc.joint_angle  # radian
        msg = Float32MultiArray()
        msg.data = self.joint_angle.tolist()
        self.segment_angle_publisher.publish(msg)

        self.event.set()
        # print(f'pub data(joint_angle) : {msg.data}')
      # else:
      #   print(f'frame does not update - flag:{self.current_frame_flag}')
  
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
          landmark_image = self.rbsc.draw_arrows(self.current_frame)
          cv2.imshow('Real-time Image with Joint Points and Arrows', landmark_image)

          if cv2.waitKey(1) & 0xFF == ord('q'):
            continue

        except Exception as e:
          self.get_logger().info(f'realtime_show() Error : {e}')
    
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
