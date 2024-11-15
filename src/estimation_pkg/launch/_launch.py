#!/usr/bin/env python3
# Copyright 2021 OROCA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command, ThisLaunchFileDir
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare
import os
from launch_ros.descriptions import ParameterValue
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    estimation_pkg_dir = get_package_share_directory('estimation_pkg')
    print(f'estimation_pkg_dir={estimation_pkg_dir}')

    rqt_perspective_file_path = os.path.join(estimation_pkg_dir, 'rqt_setting.perspective')
    print(f'rqt_perspective_file_path={rqt_perspective_file_path}')
    # rqt_command = ['rqt', '-p', rqt_perspective_file_path]

    lstm_model_path = os.path.join(estimation_pkg_dir, 'lstm_model.h5')
    scaler_x_path = os.path.join(estimation_pkg_dir, 'scaler_x.pkl')
    sclaer_y_path = os.path.join(estimation_pkg_dir, 'scaler_y.pkl')

    return LaunchDescription([
        # ExecuteProcess(
        #     cmd=rqt_command,
        #     output='screen'
        # ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [get_package_share_directory('realsense2_camera'), '/launch/rs_launch.py']),
                launch_arguments={
                'rgb_camera.color_profile': '1280,720,30',
                'depth_module.depth_profile': '1280,720,30',
                'rgb_camera.enable_auto_exposure': 'true',
                # 'rgb_camera.profile': '640,480,30',
                # 'depth_module.profile': '640,480,30',
                }.items()
        ),
        Node(
            package='estimation_pkg',
            executable='segment_angle_estimator',
            name='segment_angle_estimator',
            output='screen',
            #prefix='taskset -c 2 3'
        ),
        Node(
            package='estimation_pkg',
            executable='external_force_estimator',
            name='external_force_estimator',
            output='screen',
            parameters=[
                {'model_path': lstm_model_path},
                {'scaler_x_path': scaler_x_path},
                {'scaler_y_path': sclaer_y_path},
            ]
            #prefix='taskset -c 2 3'
        ),
        
        # Node(
        #     package='rqt_gui',
        #     executable='rqt_gui',
        #     name='rqt_gui',
        #     arguments=['--perspective-file', rqt_perspective_file_path],
        #     output='screen'
        # ),

        # 10초 후 rqt_gui 실행
        TimerAction(
            period=5.0,  # 5초 대기
            actions=[
                Node(
                    package='rqt_gui',
                    executable='rqt_gui',
                    name='rqt_gui',
                    arguments=['--perspective-file', rqt_perspective_file_path],
                    output='screen'
                )
            ]
        ),
    ])
