import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSReliabilityPolicy

from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import SetBool
from custom_interfaces.msg import LoadcellState
from custom_interfaces.msg import DataFilterSetting

from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_share_directory

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
from glob import glob
import datetime
import os
import threading
import traceback

class ExternalForceEstimationNode(Node):
  
  def __init__(self):
    super().__init__('external_force_estimation_node')

    print("============================")
    print(os.getcwd())
    print(os.path.abspath(__file__))
    print("============================")
    
    package_share_directory = get_package_share_directory('estimation_pkg')

    model_path = os.path.join(package_share_directory, 'lstm_model.h5')
    self.declare_parameter('model_path', model_path)
    self.model_path = self.get_parameter('model_path').get_parameter_value().string_value

    scaler_x_path = os.path.join(package_share_directory, 'scaler_x.pkl')
    self.declare_parameter('scaler_x_path', scaler_x_path)
    self.scaler_x_path = self.get_parameter('scaler_x_path').get_parameter_value().string_value
    
    scaler_y_path = os.path.join(package_share_directory, 'scaler_y.pkl')
    self.declare_parameter('scaler_y_path', scaler_y_path)
    self.scaler_y_path = self.get_parameter('scaler_y_path').get_parameter_value().string_value

    package_share_directory = get_package_share_directory('estimation_pkg')

    # If use _launch.py, do these lines
    self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
    self.scaler_x_path = self.get_parameter('scaler_x_path').get_parameter_value().string_value
    self.scaler_y_path = self.get_parameter('scaler_y_path').get_parameter_value().string_value


    print(f'[external_force_estimation.py] lstm_model.h5 PATH: {model_path}', flush=True)
    print(f'[external_force_estimation.py] scaler_x.pkl PATH: {scaler_x_path}', flush=True)
    print(f'[external_force_estimation.py] scaler_y.pkl PATH: {scaler_y_path}', flush=True)
    print(f'[external_force_estimation.py] lstm_model.h5 PATH: {self.model_path}', flush=True)
    print(f'[external_force_estimation.py] scaler_x.pkl PATH: {self.scaler_x_path}', flush=True)
    print(f'[external_force_estimation.py] scaler_y.pkl PATH: {self.scaler_y_path}', flush=True)

    self.declare_parameter('input_output_monitor', False)
    
    self.model = tf.keras.models.load_model(self.model_path)
    # self.model = tf.keras.models.load_model('model/lstm_model.h5')

    # 스케일러 불러오기
    self.scaler_x = joblib.load(self.scaler_x_path)
    self.scaler_y = joblib.load(self.scaler_y_path)
    # self.scaler_x = joblib.load('model/scaler_x.pkl')
    # self.scaler_y = joblib.load('model/scaler_y.pkl')

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

    self.loadcell_data_flag = False
    self.loadcell_data = LoadcellState()
    self.lc_subscriber = self.create_subscription(
        LoadcellState,
        'loadcell_state',
        self.read_loadcell_data,
        QOS_RKL1V
    )
    self.loadcell_data_offset = LoadcellState()
    self.lc_offset_subscriber = self.create_subscription(
        LoadcellState,
        'loadcell_state_offset',
        self.read_loadcell_data_offset,
        QOS_RKL1V
    )
    self.get_logger().info('loadcell_data subscriber is created.')

    
    self.wire_length_flag = False
    self.wire_length = Float32MultiArray()
    self.wire_length_subscriber = self.create_subscription(
        Float32MultiArray,
        'wire_length',
        self.read_wire_length,
        QOS_RKL1V
    )
    self.get_logger().info('wire_length subscriber is created.')
    
    self.segment_angle = Float32MultiArray()
    self.segment_angle_subscriber = self.create_subscription(
        Float32MultiArray,
        "estimated_segment_angle/absolute",
        self.segment_angle_callback,
        QOS_RKL1V)

    self.external_force = Vector3()
    self.external_force_publisher = self.create_publisher(
      Vector3,
      'estimated_external_force',
      QOS_RKL1V
    )

    self.estimatiion_thread = threading.Thread(target=self.estimation_process)
    self.estimatiion_thread.start()

  def read_fts_data(self, msg):
      self.fts_data_flag = True
      self.fts_data = msg
  def read_fts_data_offset(self, msg):
      self.fts_data_offset = msg

  def read_loadcell_data(self, msg):
      self.loadcell_data_flag = True
      self.loadcell_data = msg
  def read_loadcell_data_offset(self, msg):
      self.loadcell_data_offset = msg
  def read_wire_length(self, msg):
      self.wire_length_flag = True
      self.wire_length = msg

  def segment_angle_callback(self, msg):
    self.segment_angle = msg

  def estimation_process(self):
    """
    TODO
    LSTM force estimation code update...
    """
    self.get_logger().info(f'LSTM estimation start')
    while rclpy.ok():
      try:
        wl = self.wire_length.data
        lc = self.loadcell_data.stress
        segment_angle = self.segment_angle.data

        # self.get_logger().info(f'wl:{wl}\n lc:{lc} \n segment_angle:{segment_angle}')

        # input = np.array([ wl + lc + segment_angle])
        input = np.concatenate((wl, lc, segment_angle))
        input_for_model = input.reshape((1, 12, 1))
        predicted_normalized = self.model(input_for_model)
        predicted_denormalized = self.scaler_y.inverse_transform(predicted_normalized)

        # self.get_logger().info(f'input data: {input_for_model}')
        # self.get_logger().info(f'estimation results: {predicted_denormalized}, size:{predicted_denormalized.size}, shape:{predicted_denormalized.shape} ')

        self.external_force.x = predicted_denormalized.flatten()[0]
        self.external_force.y = predicted_denormalized.flatten()[1]
        self.external_force_publisher.publish(self.external_force)

      except Exception as e:
        #  self.get_logger().info(f'Exception: {e}')
        #  traceback.print_exc()
         pass
      finally:
         pass
    pass

"""
"""

def main(args=None):
  rclpy.init(args=args)
  try:
    estimator = ExternalForceEstimationNode()
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

