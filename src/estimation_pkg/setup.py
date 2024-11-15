from setuptools import find_packages, setup
import os, glob

package_name = 'estimation_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name), ['estimation_pkg/config.json']),
        (os.path.join('share', package_name), ['estimation_pkg/config_ROI_ref.json']),
        # (os.path.join('share', package_name, 'config'), glob('config/*.perspective')),
        (os.path.join('share', package_name), ['config/rqt_perspective/rqt_setting.perspective']),
        (os.path.join('share', package_name), ['model/lstm_model.h5']),
        (os.path.join('share', package_name), ['model/scaler_x.pkl']),
        (os.path.join('share', package_name), ['model/scaler_y.pkl']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daeyun',
    maintainer_email='bigyun9375@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
          'segment_angle_estimator = estimation_pkg.segment_angle_estimation:main',
          'external_force_estimator = estimation_pkg.external_force_estimation:main'
        ],
    },
)
