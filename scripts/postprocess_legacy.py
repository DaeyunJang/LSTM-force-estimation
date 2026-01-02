import cv2
import matplotlib.pyplot as plt

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from fil_finder import FilFinder2D
import astropy.units as u
from scipy.optimize import curve_fit, OptimizeWarning
from numpy.polynomial import Polynomial

image_path = '14_1733363817-954585449.png'


class RBSC:
    def __init__(self):
        pass

    def rotation_matrix(self, coords, theta=-90):
        rotation_matrix = np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
                                    [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]])
        return np.dot(coords, rotation_matrix.T)
    def poly4d(self, x, a, b, c, d):
        return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x
    def poly3d(self, x, a, b, c):
        return a * x ** 3 + b * x ** 2 + c * x
    def log(self, x, a, b):
        return a * np.log(b * (x + 1))
    def poly4d_sigmoid(self, x, a, b, c, d, L, k, x0):
        exp_term = np.clip(-k * (x - x0), -700, 700)
        return (a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x) + L / (1 + np.exp(exp_term))
        # return (a * x **2 + b * x) + L / (1 + np.exp(-k * (x - x0)))

    def postprocess(self, image_path, binary_thresh=100):
        # 이미지 파일 경로
        image_path = image_path

        # 1. 이미지를 그레이스케일로 읽기
        self.gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 2. 이진화
        # 임계값 설정 (예: 127)
        thresh_value = binary_thresh
        _, self.binary_image = cv2.threshold(self.gray_image, thresh_value, 255, cv2.THRESH_BINARY)

        # 연결된 컴포넌트 찾기
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.binary_image, connectivity=8)

        # 가장 큰 컴포넌트 찾기
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        # 가장 큰 컴포넌트만 남기기
        self.body_image = (labels == max_label).astype(np.uint8) * 255

        # perform skeletonization
        self.skeleton = skeletonize(self.body_image, method='lee')
        # skeleton = skeletonize(image)
        # cv2.imwrite('skeleton.png', skeleton)
        # skeleton_image = (skeleton * 255).astype(np.uint8)

        ####################################################################
        '''
        TODO
        샘플링 타임에 가장 시간 많이쓰는 구간
        없애던지 대체해야될것같기도... smoothing때문에 안해도 될것 같기도...
        '''
        # fil = FilFinder2D(self.skeleton, distance=250 * u.pc, mask=self.skeleton)
        # fil.preprocess_image(flatten_percent=85)
        # fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
        # fil.medskel(verbose=False)
        # fil.analyze_skeletons(branch_thresh=40 * u.pix, skel_thresh=10 * u.pix, prune_criteria='length')
        #
        # self.largest_backbone_image = fil.skeleton_longpath

        ####################################################################

        self.largest_backbone_image = np.copy(self.skeleton)

        # 흰색 선이 존재하는 영역을 찾기
        coords = np.column_stack(np.where(self.largest_backbone_image > 0))

        # 흰색 선이 존재하는 영역의 바운딩 박스 찾기
        x, y, w, h = cv2.boundingRect(coords)

        # 바운딩 박스를 사용하여 이미지 자르기
        self.cropped_image = self.largest_backbone_image[x:x+w, y:y+h]

        # --- 이미지 픽셀을 X,Y 데이터로 표현 ---
        # 이미지의 픽셀을 x,y좌표계로 전환
        w_star, h_star = self.cropped_image.shape[1], self.cropped_image.shape[0]
        self.xy_coords = np.column_stack(np.where(self.cropped_image > 0)).astype(np.float64)
        self.xy_coords = self.xy_coords[:, [1, 0]]    # x, y 열 위치 조정
        x_max = np.max(self.xy_coords[:, 0])
        y_max = np.max(self.xy_coords[:, 1])
        self.xy_coords[:, 1] = y_max - self.xy_coords[:, 1]    # y = height - y (이미지는 위부터 아래로 증가)

        # 폭, 높이 정규화 (이미지 사이즈 대비)
        # 더 긴쪽길이를 기준으로 축소해주어야 함.
        self.norm_xy_coords = np.copy(self.xy_coords)
        div = max(x_max, y_max)
        self.norm_xy_coords[:, 0] = self.xy_coords[:, 0] / div
        self.norm_xy_coords[:, 1] = self.xy_coords[:, 1] / div

        # 원점을 y값이 가장 작은 점으로 지정
        norm_y_min_arg = np.argmin(self.norm_xy_coords[:, 1])
        origin_xy = self.norm_xy_coords[norm_y_min_arg, :]

        # 원점을 지나는 데이터로 변경
        self.new_norm_xy_coords = self.norm_xy_coords - origin_xy

        ''' curve fitting '''
        try:
            # 커브 피팅 수행
            # test
            self.popt, _ = curve_fit(self.poly4d, self.new_norm_xy_coords[:, 0], self.new_norm_xy_coords[:, 1], maxfev=2000)
            self.fitted_y = self.poly4d(self.new_norm_xy_coords[:, 0], *self.popt)

            # main fitting
            self.trans_xy_coords = self.rotation_matrix(self.new_norm_xy_coords, theta=-90)

            self.popt_poly4d, _ = curve_fit(self.poly4d, self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], maxfev=2000)
            self.fitted_y_poly4d = self.poly4d(self.trans_xy_coords[:, 0], *self.popt_poly4d)

            self.popt_poly3d, _ = curve_fit(self.poly3d, self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], maxfev=2000)
            self.fitted_y_poly3d = self.poly3d(self.trans_xy_coords[:, 0], *self.popt_poly3d)

            self.popt_log, _ = curve_fit(self.log, self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], maxfev=2000)
            self.fitted_y_log = self.log(self.trans_xy_coords[:, 0], *self.popt_log)

            # self.popt_poly4d_sigmoid, _ = curve_fit(self.poly4d_sigmoid, self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], maxfev=2000)
            # self.fitted_y_poly4d_sigmoid = self.poly4d_sigmoid(self.trans_xy_coords[:, 0], *self.popt_poly4d_sigmoid)

        except OptimizeWarning as e:
            print(f"OptimizeWarning for image: {image_path} - returning last calculated values")
            # popt, _ = curve_fit(self.poly4d_sigmoid, self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1])

        finally:
            return self.popt_poly4d, self.popt_poly3d, self.popt_log
            # return self.popt_poly4d, self.popt_poly3d, self.popt_log, self.popt_poly4d_sigmoid

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
        plt.imshow(self.skeleton, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.title('largest backbone Image')
        plt.imshow(self.largest_backbone_image, cmap='gray')
        plt.contour(self.largest_backbone_image)
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.title('Cropped largest backbone Image')
        plt.imshow(self.cropped_image, cmap='gray')
        plt.contour(self.cropped_image)
        plt.axis('off')

        #############################################################
        plt.figure(figsize=(10,5))
        plt.subplot(2, 3, 1)
        plt.scatter(self.xy_coords[:, 0], self.xy_coords[:, 1], color='blue', s=1, label='Original data')
        plt.title('Data in pixel coordinate')
        plt.axis('scaled')
        plt.axvline(x=0, c="gray", label="x=0")
        plt.axhline(y=0, c="gray", label="y=0")
        plt.grid(linestyle='--')
        plt.xlabel('X')
        plt.ylabel('Y')

        # plt.axis('equal')

        plt.subplot(2, 3, 2)
        plt.scatter(self.norm_xy_coords[:, 0], self.norm_xy_coords[:, 1], color='black', s=1, label='Normalized data')
        plt.title('Normalized data')
        plt.axis('scaled')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-0.8, 0.8])
        plt.ylim([-0.1, 1.1])
        plt.axvline(x=0, c="gray", label="x=0")
        plt.axhline(y=0, c="gray", label="y=0")
        plt.grid(linestyle='--')

        plt.subplot(2, 3, 3)
        plt.scatter(self.new_norm_xy_coords[:, 0], self.new_norm_xy_coords[:, 1], color='black', s=0.1, label='Normalized data')
        plt.title('Normalized data (translation to origin)')
        plt.axis('scaled')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-0.8, 0.8])
        plt.ylim([-0.1, 1.1])
        plt.axvline(x=0, c="gray", label="x=0")
        plt.axhline(y=0, c="gray", label="y=0")
        plt.grid(linestyle='--')

        plt.subplot(2, 3, 4)
        plt.scatter(self.new_norm_xy_coords[:, 0], self.new_norm_xy_coords[:, 1], color='black', s=4, label='Normalized Data')
        plt.plot(self.new_norm_xy_coords[:, 0], self.fitted_y, label='Fitted Curve', color='red')
        plt.title('Curve fitting on translation to origin data')
        plt.axis('scaled')

        plt.subplot(2, 3, 5)
        plt.scatter(self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], color='green', s=1)
        plt.title('Transpose XY to YX data')
        plt.axis('scaled')

        plt.subplot(2, 3, 6)
        # plt.scatter(self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], color='green', s=4)
        plt.plot(self.trans_xy_coords[:, 0], self.fitted_y_poly4d, label='Fitted Curve', color='red')
        plt.title('Curve fitting on transpose XY to YX data')
        plt.axis('scaled')

        #############################################################
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 4, 1)
        plt.scatter(self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], color='black', s=4)
        plt.plot(self.trans_xy_coords[:, 0], self.fitted_y_poly4d, label='Fitted Curve', color='red')
        # plt.title('Poly 4d')
        plt.axis('scaled')
        text = rf'{self.popt_poly4d[0]:.2}x^4 + {self.popt_poly4d[1]:.2}x^3 + {self.popt_poly4d[2]:.2}x^2 + {self.popt_poly4d[3]:.2}x'
        plt.title(text)

        plt.subplot(1, 4, 2)
        plt.scatter(self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], color='black', s=4)
        plt.plot(self.trans_xy_coords[:, 0], self.fitted_y_poly3d, label='Fitted Curve', color='red')
        plt.title('Poly 3d')
        plt.axis('scaled')
        text = rf'{self.popt_poly3d[0]:.2}x^3 + {self.popt_poly3d[1]:.2}x^2 + {self.popt_poly3d[2]:.2}x'
        # plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=12)
        plt.title(text)

        plt.subplot(1, 4, 3)
        plt.scatter(self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], color='black', s=4)
        plt.plot(self.trans_xy_coords[:, 0], self.fitted_y_log, label='Fitted Curve', color='red')
        plt.title('Log')
        plt.axis('scaled')
        text = rf'{self.popt_log[0]:.2} log(({self.popt_log[1]:.2}x + 1))'
        # plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=12)
        plt.title(text)

        # plt.subplot(1, 4, 4)
        # plt.scatter(self.trans_xy_coords[:, 0], self.trans_xy_coords[:, 1], color='black', s=4)
        # plt.plot(self.trans_xy_coords[:, 0], self.fitted_y_poly4d_sigmoid, label='Fitted Curve', color='red')
        # plt.title('Poly 2d + Sigmoid')
        # plt.axis('scaled')
        # text = rf'${self.popt_poly4d_sigmoid[0]:.2}x^2 + {self.popt_poly4d_sigmoid[1]:.2}x + {self.popt_poly4d_sigmoid[2]:.2} / (1 + e^({-self.popt_poly4d_sigmoid[3]:.2} * (x - {self.popt_poly4d_sigmoid[4]:.2})$'
        # plt.title(text)

        plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=12)


        plt.show()

if __name__ == '__main__':
    rbsc = RBSC()
    rbsc.postprocess(image_path)
    rbsc.show()

    cv2.imwrite('image_gray.png', rbsc.binary_image)
    cv2.imwrite('image_gray_body.png', rbsc.body_image)
    cv2.imwrite('image_skeleton.png', rbsc.skeleton)
