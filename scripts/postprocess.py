import os
import sys
import glob
from xml.sax.saxutils import escape

import cv2
import matplotlib.pyplot as plt
import time
from datetime import datetime
import json
from tqdm import tqdm
from natsort import natsorted
import line_profiler
import cProfile
from numba import jit

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from fil_finder import FilFinder2D
import astropy.units as u
from scipy.optimize import curve_fit, brentq, minimize_scalar, root_scalar, OptimizeWarning
from scipy.misc import derivative
from scipy.integrate import quad, fixed_quad, quadrature
from scipy.optimize import minimize
from numpy.polynomial import Polynomial

<<<<<<< HEAD
import cProfile
from findiff import FinDiff

print(f"Current Working Directory : {os.getcwd()}")
=======
from ament_index_python.packages import get_package_share_directory
print(f"Current Working Directory : {os.getcwd()}", flush=True)
>>>>>>> 74649c1de7f436a80016c0f9e7e6fbbaa9cc26d6

image_path = '487_1723092780-883211914.png'
# image_path = '302_1723092774-710305664.png'
# image_path = '142_1723092769-375700928.png'

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")
    else:
        print(f"Directory {directory_path} already exists.")

class RBSC:
    def __init__(self, config_file='config.json'):
        self.config = self.load_config(config_file)

        self.yx_coords = None
        self.xy_coords = None
        self.__div = None

        self.func_poly4d = None
        self.func_poly3d = None
        self.func_log = None
        self.popt_poly4d = None
        self.popt_poly3d = None
        self.popt_log = None

        self.joint_yx_pixel = None
        self.joints_invtrans_xy = None
        pass

    def load_config(self, config_file):

        package_share_directory = get_package_share_directory('estimation_pkg')
        config_path = os.path.join(package_share_directory, config_file)
        print(f'[postprocess.py] config.json PATH: {config_path}', flush=True)
        
        print(f'Load {config_file}', flush=True)
        print(f'===== json list ======', flush=True)
        with open(config_path, 'r') as f:
            config = json.load(f)
            # print
            for key, value in config.items():
                print(f'{key} : {value}', flush=True)
        print(f'===== json list end ===', flush=True)

        return config

    def smoothing(self, binary, k_size=5):
        # 빈틈을 채우기 위해 닫힘 연산 적용 (모폴로지 연산)
        kernel = np.ones((k_size, k_size), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 외곽선을 찾아 근사화하지 않고 그대로 사용하여 매끄러운 곡선 유지
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 빈 이미지 생성
        smooth_image = np.zeros_like(closed)

        # 근사화되지 않은 외곽선을 따라 그리기 (매끄러운 곡선 유지)
        cv2.drawContours(smooth_image, contours, -1, 255, thickness=cv2.FILLED)

        # 경계선 스무딩을 위해 블러링 적용 후 다시 이진화
        blurred = cv2.GaussianBlur(smooth_image, (15, 15), 0)
        _, final_smooth = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        return final_smooth

    def rotation_matrix(self, coords, theta=-90):
        rotation_matrix = np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
                                    [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]])
        return np.dot(coords, rotation_matrix.T)
    def poly4d(self, x, a, b, c, d):
        return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x
        # return a * (x+2) ** 4 + b * (x+2) ** 3 + c * (x+2) ** 2 + d * (x+2)

    def poly3d(self, x, a, b, c):
        return a * x ** 3 + b * x ** 2 + c * x
    def log(self, x, a, b):
        return a * np.log(b * (x + 1))
    def poly4d_sigmoid(self, x, a, b, c, d, L, k, x0):
        exp_term = np.clip(-k * (x - x0), -700, 700)
        return (a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x) + L / (1 + np.exp(exp_term))
        # return (a * x **2 + b * x) + L / (1 + np.exp(-k * (x - x0)))

    # 곡선 양끝에 n개의 점 추가
    # y값을 기존값 그대로 쓸수도 있고, curve함수에 의해 수학적으로 구해지는 y를 쓸 수도 있다.
<<<<<<< HEAD
    def extend_curve(self, x, y, popt, rate=0.8, start_method='linear', end_method='linear', replace='origin'):
=======
    def extend_curve(self, x, y, popt, rate=0.8, start_method='linear', end_method='curve',replace='origin'):
>>>>>>> 74649c1de7f436a80016c0f9e7e6fbbaa9cc26d6
        """
        :param x: x values
        :param y: y values
        :param popt: coefficients of function
        :param rate: extend rate of number of points
        :param replace: The y value cna be used as it is (origin),
                        or the y obtained mathematically by the curve function can be used.
        :return: extended x, y values
        """
        interval =  (x.max() - x.min()) / len(x)
        n = len(x) * rate    # 전체 데이터 개수의 rate * 100 %만큼 연장
        n = int(n)
        extend_threshold = interval * n
        x_new_start = np.arange(x.min() - extend_threshold, x.min(), interval)
        x_new_end = np.arange(x.max(), x.max() + extend_threshold, interval)
        x_extended = np.concatenate([x_new_start, x, x_new_end])

        popt = popt
        if replace == 'origin':
            # default
            y_new_start = self.poly4d(x_new_start, *popt)
            y_new_end = self.poly4d(x_new_end, *popt)

            if start_method == 'linear':
                tangent = derivative(self.func_poly4d, x.min(), dx=1e-2)
                y_new_start = tangent * x_new_start
            elif start_method == 'curve':
                y_new_start = self.poly4d(x_new_start, *popt)
            if end_method == 'linear':
                tangent = derivative(self.func_poly4d, x.max(), dx=1e-2)
                y_new_end = tangent * x_new_end
            elif end_method == 'curve':
                y_new_end = self.poly4d(x_new_end, *popt)

            y_extended = np.concatenate([y_new_start, y, y_new_end])

        elif replace == 'replace':
            y_extended = self.poly4d(x_extended, *popt)

        return x_extended, y_extended

    def get_curve_length(self, x_start, x_end, popt='poly4d', method='quad', n=100):
        params = None
        if popt == 'poly4d':
            params = self.popt_poly4d
        # 피팅된 곡선 함수 정의 (curve_length 함수 내부에서)
        def fitted_curve(x):
            # return params[0] * x ** 4 + params[1] * x ** 3 + params[2] * x ** 2 + params[3] * x
            return self.poly4d(x, *params)

        # 곡선의 도함수를 수치적으로 계산하여 길이를 구함
        integrand = lambda x: np.sqrt(1 + derivative(fitted_curve, x, dx=1e-2) ** 2)
        # integrand = lambda x: np.sqrt(1 + self.dfdx_poly4d(x) ** 2)

        # length, _ = quad(integrand, x_start, x_end)
        if method == 'quad':
            length, _ = quad(integrand, x_start, x_end)
        elif method == 'quadrature':
            length, _ = quadrature(integrand, x_start, x_end)
        elif method == 'fixed_quad':
            length, _ = fixed_quad(integrand, x_start, x_end, n=n)

        return length

<<<<<<< HEAD
    # def find_x_for_given_length(self, x_start, target_length, tol=1e-2):
    #     # 길이 차이를 구하는 함수 정의
    #     def length_difference(x_end):
    #         return abs(self.get_curve_length(x_start, x_end) - target_length)
    #
    #     # 빠른 수치 미분 계산
    #     d_dx = FinDiff(0, 1e-2, 1)  # 미분간격 dx=1e-2
    #
    #     # 수치적 미분을 활용한 경사하강법이나 최적화 알고리즘 적용 가능
    #     x = x_start
    #     for _ in range(100):  # 반복 횟수 조정 가능
    #         grad = d_dx(length_difference, x)
    #         if abs(grad) < tol:
    #             break
    #         x -= length_difference(x) / grad  # Newton-Raphson과 유사한 업데이트
    #
    #     return x

    # 주어진 곡선 길이에 대해 끝점을 찾는 함수
=======
    # 주어진 곡선 길이에 대해 끝점(x1)을 찾는 함수
>>>>>>> 74649c1de7f436a80016c0f9e7e6fbbaa9cc26d6
    def find_x_for_given_length(self, x_start, target_length):
        # 곡선 길이를 계산한 후, 주어진 길이와의 차이를 반환하는 함수 정의
        def length_difference(x_end):
            return abs(self.get_curve_length(x_start, x_end, method='fixed_quad', n=100) - target_length)

        # 수치적으로 근을 찾기 위해 brentq 사용 (root-finding)
        # x_end = brentq(length_difference, x_start, x_start + target_length)  # 범위를 적절히 조절
        # return x_end

        result = minimize_scalar(
            length_difference,
            bounds=(x_start, x_start + target_length),
            method='bounded',
            options={'xatol': 1e-2}
        )
        return result.x

    def find_closest_point(self, px, py):
        # 임의의 점 (px, py)에서 곡선까지의 거리 최소화
        def distance_from_curve(x):
            return (self.func_poly4d(x) - py) ** 2 + (x - px) ** 2

        result = minimize(distance_from_curve, px)
        return result.x[0]



    def range_normalization(self, coordinates):
        # ========== Normalization of width and height based on image size ==========
        # It should be reduced based on the length of the width or height, which is longer
        x_max = np.max(coordinates[:, 0])
        y_max = np.max(coordinates[:, 1])
        self.__div = max(x_max, y_max)
        # print(f'self.__div = {self.__div}')
        self.norm_xy_coords = np.copy(coordinates)
        self.norm_xy_coords[:, 0] = self.new_xy_coords[:, 0] / self.__div
        self.norm_xy_coords[:, 1] = self.new_xy_coords[:, 1] / self.__div

        return self.norm_xy_coords

    def range_denormalization(self, coordinates):


        return self.__div, self.norm_xy_coords

    def pixel_to_orthogonal_coordinate(self, binary):
        # ========== Transform image pixel coordinate to mathematical X-Y coordinate ==========
        self.yx_coords = np.column_stack(np.where(binary > 0)).astype(np.float64)
        # self.yx_coords = self.yx_coords[:-10]
        self.num_of_pixel_points = len(self.yx_coords)
        self.xy_coords = self.yx_coords[:, [1, 0]]  # x, y 열 위치 조정
        self.xy_coords[:, 1] = binary.shape[0] - self.xy_coords[:, 1]  # height - y

        # Set the point with the smallest y to the origin
        y_min_arg = np.argmin(self.xy_coords[:, 1])
        self.origin_xy = self.xy_coords[y_min_arg, :]
        self.new_xy_coords = self.xy_coords - self.origin_xy

        self.range_normalization(self.new_xy_coords)

        return self.norm_xy_coords

    def orthogonal_to_pixel_coordinate(self, points):
        # 3. Denormalization
        self.invtrans_xy_coords = points * self.__div

        # 4. Remove the offset that was moved to the origin
        self.extended_xy_coords = self.invtrans_xy_coords + self.origin_xy

        # 5. put back mathematical X-Y coordinate to image coordinate
        x = np.around(self.extended_xy_coords[:, 0])
        y = np.around(self.extended_xy_coords[:, 1])
        new_x, new_y = x.astype(int), y.astype(int)
        new_y = self.image_h - new_y
        # exceptions for values above image size
        valid_indices = (new_x >= 0) & (new_x < self.image_w) & (new_y >= 0) & (new_y < self.image_h)
        new_x = new_x[valid_indices]
        new_y = new_y[valid_indices]
        self.extended_curve = np.zeros((self.image_h, self.image_w))
        # get new extended skeleton curve
        self.extended_curve[new_y, new_x] = 1
        self.extended_skeleton_origin_yx = np.column_stack(np.where(self.extended_curve > 0)).astype(np.float64)

        ext_curve = np.zeros((self.image_h, self.image_w))
        ext_curve[new_y,new_x] = 1
        yx_pixel = np.column_stack(np.where(ext_curve > 0)).astype(np.float64)

        return yx_pixel

    def get_joints(self):
        # ============= find junction points ============
        # 끝점의 기울기 - test
        # self.tip_tangent = derivative(self.func_poly4d, self.x1, dx=1e-2)
        # self.tip_angle_rad = np.arctan(self.tip_tangent)
        # self.tip_angle_deg = np.degrees(self.tip_angle_rad)
        # print(f'tip_angle (deg) = {self.tip_angle_deg}')

        ###########################################################################
        # get each angle of joints
        num_of_segments = self.config.get('num_of_segments')
        length_of_segment = self.config.get('length_of_segment')  # mm
        self.joint_x = np.zeros(num_of_segments)
        # print('============ get joints_x ================')
        # print(f'total len = {self.curve_length}')

        target_len = self.curve_length / float(2 * num_of_segments)
        
        x1 = self.find_x_for_given_length(self.start_point[0], target_length=target_len)
        self.joint_x[0] = x1

        for i in range(1, num_of_segments):
            target_len = self.curve_length / float(num_of_segments)
            x1 = self.find_x_for_given_length(self.joint_x[i-1], target_length=target_len)
            self.joint_x[i] = x1
            # print(f'i={i} || target_len={target_len} | x0={self.joint_x[i-1]} -> x1={self.joint_x[i]}')

        ###########################################################################
        # self.joint_y = self.poly4d(self.joint_x, *self.popt_poly4d)
        self.joint_y = self.func_poly4d(self.joint_x)
        self.joints_xy = np.column_stack((self.joint_x, self.joint_y))

        self.joint_tangents = np.array([derivative(self.func_poly4d, x0, dx=1e-2) for x0 in self.joint_x])
        self.joint_angle = np.arctan(self.joint_tangents)
        self.joint_angle_degree = np.degrees(self.joint_angle)

        # first(0) joint is unstable compared to another joints
        # 첫번째 segment는 오차가 있어서 점사이의 벡터를 가지고 각도를 따로 정의함
        vector_seg1_to_seg2 = [self.joints_xy[1,0]-self.joints_xy[0,0], self.joints_xy[1,1]-self.joints_xy[0,1]]
        tangent = vector_seg1_to_seg2[1] / vector_seg1_to_seg2[0]
        theta = np.arctan(tangent)
        joint_0_theta = theta/2.0
        self.joint_angle[0] = joint_0_theta
        self.joint_angle_degree[0] = np.degrees(self.joint_angle[0])

        diff_joint_angle = np.diff(self.joint_angle)
        self.joint_angle_relative = np.insert(diff_joint_angle, 0, self.joint_angle[0])
        self.joint_angle_relative_degree = np.degrees(self.joint_angle_relative)

        # print(f'joints = {self.joints_xy}')
        # print(f'joint_tangent = {self.joint_tangents}')
        # print(f'joint_angle (rad) = {self.joint_angle}')
        # print(f'joint_angle (deg) = {self.joint_angle_degree}')

        '''
        @ TODO
        have to add the operation for estimating position of each joints
        using each estimated angles

        '''

    def postprocess(self, image, binary_thresh=127, filfinder_flag=False):
        try:
            # ========== post processing ==========
            # 1. read as grayscale
            self.image = image
            if image is None or image.size == 0:
                # print("Error: image : {image}")
                return
            self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # self.gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            self.image_w, self.image_h = self.gray_image.shape[1], self.gray_image.shape[0]

            # 2. Binarization
            # 임계값 설정 (예: 127)
            thresh_value = binary_thresh
            _, self.binary_image = cv2.threshold(self.gray_image, thresh_value, 255, cv2.THRESH_BINARY)

            # 3. Find connected commponets
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.binary_image, connectivity=8)
            if num_labels < 1:
                return

            # 4. Find the largest componnet
            max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

            # 5. Leave only the largest commpent (it is body)
            self.body_image_origin = (labels == max_label).astype(np.uint8) * 255
            self.body_image = self.smoothing(binary=self.body_image_origin, k_size=15)

            # print(f'{self.image} / {self.image.shape} / {self.image.size} ')
            # print(f'{self.body_image} / {self.body_image.shape} / {self.body_image.size} ')
            # ========== Find skeleton of backbone ==========
            # perform skeletonization
            self.skeleton = skeletonize(self.body_image, method='lee')

            ####################################################################
            '''
            TODO
            샘플링 타임에 가장 시간 많이쓰는 구간
            없애던지 대체해야될것같기도... smoothing때문에 안해도 될것 같기도...
            '''
            # Find the longest curve of backbone
            if filfinder_flag == True:
                fil = FilFinder2D(self.skeleton, distance=250 * u.pc, mask=self.skeleton)
                fil.preprocess_image(flatten_percent=85)
                fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
                fil.medskel(verbose=False)
                fil.analyze_skeletons(branch_thresh=40 * u.pix, skel_thresh=30 * u.pix, prune_criteria='length')
                self.longest_backbone_image = fil.skeleton_longpath
            else:
                self.longest_backbone_image = np.copy(self.skeleton)
            ####################################################################

            self.pixel_to_orthogonal_coordinate(self.longest_backbone_image)
        except Exception as e:
            print(f'postprocess error : {e}', flush=True)
            return
        
        # ========== Curve fitting ==========
        try:
            # 데이터 가중치 설정
            # weights = np.ones_like(self.yx_coords[:, 0])
            # sigma = 0.1
            # num = 10
            # weights[0:num] = sigma
            # weights[-num:] = sigma
            # sigma = 1 / weights

            # test
            self.popt, _ = curve_fit(self.poly4d, self.norm_xy_coords[:, 0], self.norm_xy_coords[:, 1], maxfev=2000)
            self.fitted_y = self.poly4d(self.norm_xy_coords[:, 0], *self.popt)

            # 1. Rotate the points clockwise 90 degree
            self.trans_xy_coords = self.rotation_matrix(self.norm_xy_coords, theta=-90)

            # 2. curve fitting
            self.temp_popt_poly4d, _ = curve_fit(self.poly4d, self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], maxfev=2000)
            # self.popt_poly4d, _ = curve_fit(self.poly4d, self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], sigma=sigma, maxfev=2000)
            self.fitted_y_poly4d = self.poly4d(self.trans_xy_coords[:, 0], *self.temp_popt_poly4d)
            self.func_poly4d = lambda x: self.poly4d(x, *self.temp_popt_poly4d)
            self.dfdx_poly4d = lambda x: self.temp_popt_poly4d[0]*4*x**3 + self.temp_popt_poly4d[1]*3*x**2 + self.temp_popt_poly4d[2]*2*x + self.temp_popt_poly4d[3]
            # self.popt_poly3d, _ = curve_fit(self.poly3d, self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], maxfev=2000)
            # self.fitted_y_poly3d = self.poly3d(self.trans_xy_coords[:, 0], *self.popt_poly3d)
            #
            # self.popt_log, _ = curve_fit(self.log, self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], maxfev=2000)
            # self.fitted_y_log = self.log(self.trans_xy_coords[:, 0], *self.popt_log)

            # self.popt_poly4d_sigmoid, _ = curve_fit(self.poly4d_sigmoid, self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], maxfev=2000)
            # self.fitted_y_poly4d_sigmoid = self.poly4d_sigmoid(self.trans_xy_coords[:, 0], *self.popt_poly4d_sigmoid)

            # ============================================================
            ''' 
            TODO
            Extrapolation for getting full center line
            1. find startpoint(endpoint) of skeleton
            2. set startpoint as origin (get offset of x, y)
            3. curvefit
            4. put back origin to startpoint in 2.
            5. extrapolation 
            6. get center line of object
            '''
            # 1. Extraploation
            ext_coords_x, ext_coords_y = self.extend_curve(self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], self.temp_popt_poly4d, rate=0.2)

            # Import
            # update curvefit coefficients
            """
            TODO
            곡선 연장후 재피팅 안해줄 경우 skeleton 끝이 테두리로 삐져나갔을 때 대처가 안됌
            """
            self.popt_poly4d, _ = curve_fit(self.poly4d, ext_coords_x, ext_coords_y, maxfev=2000)
            self.fitted_y_poly4d = self.poly4d(ext_coords_x, *self.popt_poly4d)
            self.func_poly4d = lambda x: self.poly4d(x, *self.popt_poly4d)

            self.extended_norm_coords = np.column_stack((ext_coords_x, ext_coords_y))

            # 2. Inverse transformation : rotate the points CCW 90 degree
            self.invtrans_norm_xy_coords = self.rotation_matrix(self.extended_norm_coords, theta=90)

            self.orthogonal_to_pixel_coordinate(self.invtrans_norm_xy_coords)

            # Remove curve points outside the boundary
            body_image = np.copy(self.body_image) / 255.0
            self.extended_skeleton = np.logical_and(body_image, self.extended_curve).astype(int)
            self.extended_yx_coords = np.column_stack(np.where(self.extended_skeleton > 0)).astype(np.float64)
            self.extended_xy_coords = self.extended_yx_coords[:, [1, 0]] # x, y 열 위치 조정
            self.extended_xy_coords[:, 1] = self.extended_skeleton.shape[0] - self.extended_xy_coords[:, 1]

            self.new_extended_xy_coords = self.extended_xy_coords - self.origin_xy

            # ========== Normalization of width and height based on image size ==========
            self.norm_extended_xy_coords = np.copy(self.new_extended_xy_coords)
            self.norm_extended_xy_coords[:, 0] = self.new_extended_xy_coords[:, 0] / self.__div
            self.norm_extended_xy_coords[:, 1] = self.new_extended_xy_coords[:, 1] / self.__div

            # find start and end point
            self.norm_trans_extended_xy_coords = self.rotation_matrix(self.norm_extended_xy_coords, theta=-90)
            x_max_arg = np.argmax(self.norm_trans_extended_xy_coords[:, 0])
            x_min_arg = np.argmin(self.norm_trans_extended_xy_coords[:, 0])
            self.end_point = self.norm_trans_extended_xy_coords[x_max_arg, :]
            self.start_point = self.norm_trans_extended_xy_coords[x_min_arg, :]

            # define start and end point
            self.x0 = self.find_closest_point(self.start_point[0], self.start_point[1])
            self.y0 = self.func_poly4d(self.x0)
            self.x1 = self.find_closest_point(self.end_point[0], self.end_point[1])
            self.y1 = self.func_poly4d(self.x1)

            # get total length of function in range
            self.curve_length = self.get_curve_length(self.x0, self.x1)

            # ============= find junction points ============
            self.get_joints()

            self.joints_invtrans_xy = self.rotation_matrix(self.joints_xy, theta=90)
            self.joint_yx_pixel = self.orthogonal_to_pixel_coordinate(self.joints_invtrans_xy)

            self.image_rgb_with_landmarks = self.draw_arrows(self.image)

            a=0

        except Exception as e:
            print(f"postprocess() error : {e}", flush=True)
            # print(f"Error : {e} \n OptimizeWarning for image: {image_path} - returning last calculated values", flush=True)
            pass

        finally:
            return self.popt_poly4d, self.popt_poly3d, self.popt_log

    def draw_arrows(self, image):
        try:
            # Compute arrow vectors
            frame = image
            rads = self.joint_angle + np.pi / 2
            arrow_length = 15
            u = np.cos(rads) * arrow_length
            v = np.sin(rads) * arrow_length
            v = -v  # Reverse v to match the original behavior

            # Draw joint points and arrows
            pixel_yx = np.flip(self.joint_yx_pixel, axis=0)
            for point in pixel_yx:
                cv2.circle(frame, (int(point[1]), int(point[0])), 2, (0, 0, 255), -1)  # Red dots for joints

            for i in range(len(u)):
                start_point = (int(pixel_yx[i, 1]), int(pixel_yx[i, 0]))  # yx needs to be flipped for cv2
                end_point = (int(start_point[0] + u[i]), int(start_point[1] + v[i]))  # u and v provide direction
                # Draw the arrow line for each point
                cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 2, tipLength=0.3)

            return frame
        except Exception as e:
            # print(f'draw_arrows() error: {e}', flush=True)
            return

    def plot_save(self, save_dir):
        # results_0.4mm
        # rads = -np.deg2rad(self.joint_angle_degree)
        rads = self.joint_angle + np.pi / 2
        # print(f'rads = {rads}')
        arrow_length = 15
        u = np.cos(rads) * arrow_length
        v = np.sin(rads) * arrow_length
        v = -v
        vector_scale = 1
        plt.figure(figsize=(10, 5))
        # plt.subplot(2, 3, 6)
        plt.title('Grayscale Image')
        plt.imshow(self.gray_image, cmap='gray')
        pixel_yx = np.flip(self.joint_yx_pixel, axis=0)
        plt.scatter(pixel_yx[:, 1], pixel_yx[:, 0], color='red', s=4, label='Joint Point')
        plt.quiver(pixel_yx[:, 1], pixel_yx[:, 0],
                   u, v,
                   angles='xy',
                   scale_units='xy',
                   scale=vector_scale,
                   width=0.005,
                   headwidth=2,
                   headlength=2,
                   headaxislength=2,
                   color='blue'
                   )
        plt.axis('off')

        # 이미지를 파일로 저장
        dir_name = save_dir
        image_filename = os.path.basename(self.image_path)
        plt.savefig(os.path.join(dir_name, image_filename), bbox_inches='tight', pad_inches=0)

    def show(self):
        # 결과 출력
        plt.figure(figsize=(10, 5))

        # 그레이스케일 이미지 출력
        plt.subplot(2, 3, 1)
        plt.title('Grayscale Image')
        plt.imshow(self.gray_image, cmap='gray')
        plt.axis('off')

        # 이진화 이미지 출력
        plt.subplot(2, 3, 2)
        plt.title('Binary Image')
        plt.imshow(self.binary_image, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.title('Body-only Image')
        plt.imshow(self.body_image, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.title('Skeleton Image')
        # plt.imshow(self.skeleton, cmap='jet', alpha=0.5)
        plt.imshow(self.body_image, cmap='gray', alpha=1)
        plt.scatter(self.yx_coords[:, 1], self.yx_coords[:, 0], color='blue', s=1, label='Original skeleton')
        # plt.contour(self.longest_backbone_image)
        plt.legend()
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(self.body_image, cmap='gray')
        # plt.scatter(self.extended_curve[:, 1], self.extended_curve[:, 0], color='red', s=10)
        plt.scatter(self.extended_yx_coords[:, 1], self.extended_yx_coords[:, 0], color='orange', s=4, label='Extended skeleton')
        plt.scatter(self.yx_coords[:, 1], self.yx_coords[:, 0], color='blue', s=1, label='Original skeleton')
        plt.legend()
        plt.title('Extended Pixel')
        plt.axis('scaled')

        plt.subplot(2, 3, 6)
        plt.title('Extended skeleton Image')
        plt.imshow(self.body_image, cmap='gray', alpha=1)
        plt.scatter(self.extended_yx_coords[:, 1], self.extended_yx_coords[:, 0], color='orange', s=1,
                    label='Extended skeleton')
        plt.scatter(self.extended_skeleton_origin_yx[:, 1], self.extended_skeleton_origin_yx[:, 0], color='red', s=4,
                    label='joints')
        plt.legend()
        plt.axis('off')


        # results_0.4mm
        # rads = -np.deg2rad(self.joint_angle_degree)
        rads = self.joint_angle + np.pi/2
        print(f'rads = {rads}', flush=True)
        arrow_length = 15
        u = np.cos(rads) * arrow_length
        v = np.sin(rads) * arrow_length
        v = -v
        vector_scale = 1
        plt.figure(figsize=(10,5))
        # plt.subplot(2, 3, 6)
        plt.title('Grayscale Image')
        plt.imshow(self.gray_image, cmap='gray')
        pixel_yx = np.flip(self.joint_yx_pixel, axis=0)
        plt.scatter(pixel_yx[:, 1], pixel_yx[:, 0], color='red', s=4, label='Joint Point')
        plt.quiver(pixel_yx[:, 1], pixel_yx[:, 0],
                   u, v,
                   angles='xy',
                   scale_units='xy',
                   scale=vector_scale,
                   width=0.005,
                   headwidth=2,
                   headlength=2,
                   headaxislength=2,
                   color='blue'
                   )
        plt.axis('off')

        #############################################################
        plt.figure(figsize=(10,5))
        plt.subplot(3, 3, 1)
        plt.scatter(self.xy_coords[:, 0], self.xy_coords[:, 1], color='black', s=1, label='Original data')
        plt.title('Data unit pixel size')
        plt.axis('scaled')
        # plt.axis('equal')
        plt.grid(True)

        plt.subplot(3, 3, 2)
        plt.scatter(self.new_xy_coords[:, 0], self.new_xy_coords[:, 1], color='black', s=1, label='Normalized data')
        plt.title('Translation to origin data')
        plt.axis('scaled')
        plt.grid(True)

        plt.subplot(3, 3, 3)
        plt.scatter(self.norm_xy_coords[:, 0], self.norm_xy_coords[:, 1], color='black', s=1, label='Normalized data')
        plt.title('Normalized data')
        plt.axis('scaled')
        plt.grid(True)

        plt.subplot(3, 3, 4)
        plt.scatter(self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], color='black', s=1)
        plt.title('Transpose XY to YX data')
        plt.axis('scaled')
        plt.grid(True)

        plt.subplot(3, 3, 5)
        plt.scatter(self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], color='black', s=4)
        plt.scatter(self.extended_norm_coords[:, 0], self.fitted_y_poly4d, color='red', s=1)
        text = rf'{self.popt_poly4d[0]:.2}x^4 + {self.popt_poly4d[1]:.2}x^3 + {self.popt_poly4d[2]:.2}x^2 + {self.popt_poly4d[3]:.2}x'
        plt.title(text)
        plt.axis('scaled')
        plt.grid(True)

        #############################################################
        x_range = np.linspace(self.start_point[0], self.end_point[0], 100)

        # 시각화
        plt.figure(figsize=(8, 6))
        plt.plot(self.norm_trans_extended_xy_coords[:, 0], self.norm_trans_extended_xy_coords[:, 1], 'yo', markersize=10,
                 label='Extended Data(Skeleton) Points')
        plt.plot(self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], 'bo', markersize=5, label='Origin Data(Skeleton) Points')
        plt.plot(x_range, self.func_poly4d(x_range), 'k-', label='Fitted Curve')
        plt.plot(self.start_point[0], self.start_point[1], 'ro', label=f'Measured Start Point ({self.start_point[0]:.5f}, {self.start_point[1]:.5f})')
        plt.plot([self.end_point[0]], [self.end_point[1]], 'ro', label=f'Measured End Point ({self.end_point[0]:.5f}, {self.end_point[1]:.5f})')
        plt.plot([self.x0], [self.y0], 'gx', label=f'Start Closest Point ({self.x0:.5f}, {self.y0:.5f})')
        plt.plot([self.x1], [self.y1], 'gx', label=f'End Closest Point ({self.x1:.5f}, {self.y1:.5f})')
        plt.plot(self.joint_x, self.joint_y, 'ro', label=f'Joint Point')
        # plt.legend(loc='upper left', bbox_to_anchor=(0, -0.2), ncol=1)
        plt.legend()
        plt.title('Curve Fitting and Closest Point Visualization')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('scaled')
        plt.grid(True)

        #############################################################
        plt.figure(figsize=(8, 6))
        # plt.subplot(1, 3, 2)
        plt.plot(x_range, self.func_poly4d(x_range), 'k-', label='Fitted Curve')
        plt.plot(self.joint_x, self.joint_y, 'ro', label=f'Joint Point')
        plt.plot(self.start_point[0], self.start_point[1], 'bo',
                 label=f'Measured Start Point ({self.start_point[0]:.5f}, {self.start_point[1]:.5f})')
        plt.plot([self.end_point[0]], [self.end_point[1]], 'bo',
                 label=f'Measured End Point ({self.end_point[0]:.5f}, {self.end_point[1]:.5f})')

        # plt.legend(loc='upper right', bbox_to_anchor=(0, -0.2), ncol=1)
        plt.legend()
        plt.title('Curve Fitting and Closest Point Visualization (Invers-Rot-Transform)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True)

        #############################################################

        plt.show()

if __name__ == '__main__':
    rbsc = RBSC()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rbsc.postprocess(image)
    rbsc.show()

    # img_dir = 'data/2024-08-08 experiment/2024-08-08-13-52-44 nopayload/images'
<<<<<<< HEAD
    # dir_path = '../data/2024-08-08 experiment/2024-08-08-13-46-11 upper_init-X'
    # img_dir = os.path.join(dir_path, 'images')
    # images = [img for img in os.listdir(img_dir) if img.endswith(".png") or img.endswith(".jpg")]
    # images = natsorted(images)
    #
    # # 현재 시간 가져오기
    # current_time = datetime.now()
    # time_str = current_time.strftime("%Y%m%d_%H%M%S")
    #
    # dir_name = os.path.join(dir_path, f'images_with_arrow_{time_str}')
    # create_directory(dir_name)
    # for image_name in tqdm(images):
    #     image_path = os.path.join(img_dir, image_name)
    #     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #     rbsc.postprocess(image)
    #     rbsc.plot_save(dir_name, image_name)
=======
    img_dir = 'data/2024-08-08 experiment/2024-08-08-13-46-11 upper_init-X/images'
    images = [img for img in os.listdir(img_dir) if img.endswith(".png") or img.endswith(".jpg")]
    images = natsorted(images)

    # 현재 시간 가져오기
    current_time = datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")

    dir_name = f'images_with_arrow_{time_str}'
    create_directory(dir_name)
    for image_name in tqdm(images):
        image_path = os.path.join(img_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        rbsc.postprocess(image)
        rbsc.plot_save(dir_name)
>>>>>>> 74649c1de7f436a80016c0f9e7e6fbbaa9cc26d6

    # sampling time test
    n = 1
    st = time.time()
    for i in tqdm(range(n)):
        # cProfile.run('rbsc.postprocess(image_path)')
        cProfile.run('rbsc.postprocess(image)')
        # rbsc.postprocess(image)
        # rbsc.plot_save()
        # print(f'num of sequence = {i}')
    et = time.time()
    sampling_freq = n/(et-st)
    print(f'total time : {et-st} / freq = {sampling_freq}', flush=True)

    # cv2.imwrite('image_gray.png', rbsc.binary_image)
    # cv2.imwrite('image_gray_body.png', rbsc.body_image)
    # cv2.imwrite('image_skeleton.png', rbsc.skeleton)
