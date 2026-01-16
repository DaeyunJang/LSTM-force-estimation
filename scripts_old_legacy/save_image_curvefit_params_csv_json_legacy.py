import os
import cv2
import numpy as np
import pandas as pd
import json
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from natsort import natsorted

# custom moudle
from postprocess import RBSC

############################################################################

# 디렉토리 생성 함수
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")
    else:
        print(f"Directory {directory_path} already exists.")

folder_path = os.path.join('../data/2024-08-08 experiment/2024-08-08-13-59-06 upper_init-O')
image_path = os.path.join(folder_path, 'images')
# 결과를 저장할 디렉토리 생성
output_directory_name_body = 'images_crop_body'
output_directory_name_body = os.path.join(folder_path, output_directory_name_body)
create_directory(output_directory_name_body)

output_directory_name_skeleton = 'images_skeleton'
output_directory_name_skeleton = os.path.join(folder_path, output_directory_name_skeleton)
create_directory(output_directory_name_skeleton)

output_directory_name_lb = 'images_longest_backbone'
output_directory_name_lb = os.path.join(folder_path, output_directory_name_lb)
create_directory(output_directory_name_lb)

result_dir = 'results'
result_dir = os.path.join(folder_path, result_dir)
create_directory(result_dir)
############################################################################

def low_pass_filter(data, cutoff_frequency=0.2, sampling_rate=30.0, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data
############################################################################

def process(dir_path, rbsc: RBSC, do_LPF=True, do_curvefit=True):
    '''
    :param dir_path:
    :param rbsc:
    :param do_LPF:
    :param do_curvefit:
    :return:
    @note
        1. Low-Pass Filter and adding normalized each column data
        2. Image post processing (curve fit)
        3. Save fitting parameters as .csv and .json, respectly.
    '''
    if do_LPF == True:
        ## save LPF filtered .csv file
        csv_path = os.path.join(folder_path, 'data.csv')
        data = pd.read_csv(csv_path)
        columns_to_filter = ['loadcell #0', 'loadcell #1', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']

        for column in columns_to_filter:
            filtered_values = low_pass_filter(data[column].values)
            data[column] = filtered_values

        scaler = MinMaxScaler()
        columns_to_normalized = ['wire length #0', 'wire length #1', 'loadcell #0', 'loadcell #1', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']
        for column in columns_to_normalized:
            # adding normalized data
            filtered_values = low_pass_filter(data[column].values)
            normalized_column_name = column + '_normalized'
            data[normalized_column_name] = scaler.fit_transform(filtered_values.reshape(-1, 1)).flatten()

        # 필터링된 데이터를 새로운 CSV 파일로 저장
        filtered_file_path = os.path.join(folder_path, 'data_LPF.csv')  # 필터링된 데이터 저장할 새로운 CSV 파일 경로

        # 파일이 존재할 경우 사용자에게 확인 요청
        if os.path.exists(filtered_file_path):
            response = input(f"File {filtered_file_path} already exists. Do you want to overwrite it? (y/n): ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                exit()

        data.to_csv(filtered_file_path, index=False)
        print(f"Filtered data saved to {filtered_file_path}")

    if do_curvefit == True:
        # 이미지를 순차적으로 읽고 fitting 결과를 저장
        csv_results_poly4d = []
        csv_results_poly3d = []
        csv_results_log = []
        csv_results_joint_angle = []

        json_results_poly4d = []
        json_results_poly3d = []
        json_results_log = []
        json_results_joint_angle = []

        list = os.listdir(dir_path)
        # sorted_list = sorted(list, key=lambda x: int(os.path.splitext(x)[0]))
        # sorted_list = sorted(list)
        sorted_list = natsorted(list)
        for filename in tqdm(sorted_list):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(dir_path, filename)
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                coef_4d, coef_3d, coef_log = rbsc.postprocess(image)
                joint_angle = rbsc.joint_angle
                if coef_4d is not None:
                    csv_results_poly4d.append((filename, coef_4d))
                    json_results_poly4d.append({'Image Filename': filename, 'Coefficients': coef_4d.tolist()})
                if coef_3d is not None:
                    csv_results_poly3d.append((filename, coef_3d))
                    json_results_poly3d.append({'Image Filename': filename, 'Coefficients': coef_3d.tolist()})
                if coef_log is not None:
                    csv_results_log.append((filename, coef_log))
                    json_results_log.append({'Image Filename': filename, 'Coefficients': coef_log.tolist()})

                csv_results_joint_angle.append((filename, joint_angle))
                json_results_joint_angle.append({'Image Filename': filename, 'Coefficients': joint_angle.tolist()})

                img_save_path = os.path.join(output_directory_name_body, filename)
                cv2.imwrite(img_save_path, rbsc.body_image)
                img_save_path = os.path.join(output_directory_name_skeleton, filename)
                cv2.imwrite(img_save_path, rbsc.skeleton.astype(np.uint8)*255)
                img_save_path = os.path.join(output_directory_name_lb, filename)
                cv2.imwrite(img_save_path, rbsc.longest_backbone_image.astype(np.uint8)*255)

        # 결과를 데이터프레임으로 저장
        # csv
        df0 = pd.DataFrame(csv_results_poly4d, columns=['Image Filename', 'Coefficients'])
        df1 = pd.DataFrame(csv_results_poly3d, columns=['Image Filename', 'Coefficients'])
        df2 = pd.DataFrame(csv_results_log, columns=['Image Filename', 'Coefficients'])
        df3 = pd.DataFrame(csv_results_joint_angle, columns=['Image Filename', 'Joint Angles'])

        df0.to_csv(result_dir + '/' + 'curve_fit_result-poly4d.csv', index=False)
        df1.to_csv(result_dir + '/' + 'curve_fit_result-poly3d.csv', index=False)
        df2.to_csv(result_dir + '/' + 'curve_fit_result-log.csv', index=False)
        df3.to_csv(result_dir + '/' + 'curve_fit_result-joint_angle.csv', index=False)
        print(f".csv file saved.")

        # json
        with open(result_dir + '/' + 'curve_fit_result-poly4d.json', 'w') as f:
            json.dump(json_results_poly4d, f)
            print(f".json file saved to {result_dir + '/' + 'curve_fit_result-poly4d.json'}.")
        with open(result_dir + '/' + 'curve_fit_result-poly3d.json', 'w') as f:
            json.dump(json_results_poly3d, f)
            print(f".json file saved to {result_dir + '/' + 'curve_fit_result-poly3d.json'}.")
        with open(result_dir + '/' + 'curve_fit_result-log.json', 'w') as f:
            json.dump(json_results_log, f)
            print(f".json file saved to {result_dir + '/' + 'curve_fit_result-log.json'}.")
        with open(result_dir + '/' + 'curve_fit_result-joint_angle.json', 'w') as f:
            json.dump(json_results_joint_angle, f)
            print(f".json file saved to {result_dir + '/' + 'curve_fit_result-joint_angle.json'}.")

        return df0, df1, df2
        # return df0, df1, df2, df3

# 이미지 프로세싱 클래스
rbsc = RBSC()
# 폴더 처리 및 결과 저장
df_results = process(image_path, rbsc)

# 결과 출력
print(df_results)
