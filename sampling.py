"""
sampling.py
Zhiang Chen
Sept 2020
"""

import numpy as np
import PIL.Image
import cv2
from sklearn.neighbors import KDTree


class SP(object):
    def __init__(self):
        pass

    def readOriginal(self, detection_image, class_image, mask_image):
        self.det_img = np.array(PIL.Image.open(detection_image))
        self.clas_img = np.array(PIL.Image.open(class_image))
        self.mask_img = np.array(PIL.Image.open(mask_image))[:, :, 0]
        self.U, self.V = self.det_img.shape

    def grid_images(self, size):
        self.size = size
        step_u = int(self.U/size)
        step_v = int(self.V/size)
        self.grid = np.zeros((size, size, 3))  # detection, classification, and mask
        for u in range(size):
            for v in range(size):
                detect_tile = self.det_img[u*step_u: u*step_u+step_u, v*step_v: v*step_v+step_v]
                clas_tile = self.clas_img[u * step_u: u * step_u + step_u, v * step_v: v * step_v + step_v]
                mask_tile = self.mask_img[u * step_u: u * step_u + step_u, v * step_v: v * step_v + step_v]
                self.grid[u, v, 0] = int(detect_tile.max()/50)
                self.grid[u, v, 1] = clas_tile.max() - 1 if clas_tile.max() != 0 else 0
                self.grid[u, v, 2] = mask_tile.max()
        detect = self.grid[:, :, 0]
        clas = self.grid[:, :, 1]
        mask = self.grid[:, :, 2]
        cv2.imwrite('detect_grid.png', detect)
        cv2.imwrite('clas_grid.png', clas)
        cv2.imwrite('mask_grid.png', mask)

    def sample(self):
        U, V = self.grid[:, :, 0].shape
        detect_train_data = np.zeros((U, V))
        for u in range(U):
            for v in range(V):
                detect_index = self.grid[u, v, 0]  # 0(no data), 1(no damage), 2(EF0), 3(EF1), 4(EF2), 5(EF3)
                clas_index = self.grid[u, v, 1]  # 0(no data), 1(no damage), 2(light damage), 3(severe damage)
                if detect_index == 0:
                    detect_train_data[u, v] = 0
                elif detect_index == 1:
                    detect_train_data[u, v] = 1 if clas_index <= 1 else 1.5
                elif detect_index == 2:
                    if clas_index == 0:
                        detect_train_data[u, v] = 2
                    elif clas_index == 1:
                        detect_train_data[u, v] = 1.5
                    elif clas_index == 2:
                        detect_train_data[u, v] = 2
                    else:
                        detect_train_data[u, v] = 2.5
                elif detect_index == 3:
                    if clas_index == 0:
                        detect_train_data[u, v] = 3
                    elif clas_index == 1:
                        detect_train_data[u, v] = 2.5
                    elif clas_index == 2:
                        detect_train_data[u, v] = 3
                    else:
                        detect_train_data[u, v] = 3.5
                elif detect_index == 4:
                    if clas_index == 0:
                        detect_train_data[u, v] = 4
                    elif clas_index <= 2:
                        detect_train_data[u, v] = 3.5
                    else:
                        detect_train_data[u, v] = 4
                elif detect_index == 5:
                    if clas_index == 0:
                        detect_train_data[u, v] = 5
                    elif clas_index <= 2:
                        detect_train_data[u, v] = 3.5
                    else:
                        detect_train_data[u, v] = 5
        cv2.imwrite("resampled_detection.png", detect_train_data*50)
        clas_train_data = np.zeros((U, V))
        # split the grid further
        detect_grid = self.grid[:, :, 0]
        clas_grid = self.grid[:, :, 1]
        # build a KD tree for detect_grid
        detect_grid_indices = np.array(np.nonzero(detect_grid)).transpose()
        #print(detect_grid_indices)
        kd_tree = KDTree(detect_grid_indices)
        # sample clas_train_data
        step = int(self.size/20)
        for i in range(20):  # grid number along U
            for j in range(20):  # grid number along V
                detect_tile = detect_grid[i*step:i*step+step, j*step:j*step+step]
                clas_tile = clas_grid[i * step:i * step + step, j * step:j * step + step]
                if clas_tile.min() > 0:
                    detect_nm = np.count_nonzero(detect_tile)
                    threshold_nm = 20
                    #if detect_nm < threshold_nm:  # a threshold for checking existing detect numbers
                    query_id = np.array((i*step+step/2, j*step+step/2)).reshape(1, 2)
                    nearest_dist, nearest_id = kd_tree.query(query_id, k=5)
                    nearest_detect_grid_indices = detect_grid_indices[nearest_id].reshape(-1, 2)
                    """
                    print(nearest_detect_grid_indices)
                    print(query_id)
                    print(nearest_id)
                    print('\n')
                    """
                    nearest_detect_values = np.array([detect_grid[tuple(i)] for i in nearest_detect_grid_indices])
                    nearest_detect_values_mean = nearest_detect_values.mean()
                    clas_u = np.random.rand(threshold_nm)  # the sample should at least reach the threshold
                    sample_clas_u = (clas_u*step).astype(int)
                    clas_v = np.random.rand(threshold_nm)  # the sample threshold
                    sample_clas_v = (clas_v * step).astype(int)
                    for sample_id in range(threshold_nm):  # the sample threshold
                        u = sample_clas_u[sample_id] + i*step
                        v = sample_clas_v[sample_id] + j*step
                        clas_value = clas_grid[u, v]
                        clas_train_data[u, v] = self.decide_clas_value(clas_value, nearest_detect_values_mean)
        cv2.imwrite("resampled_classification.png", clas_train_data * 50)
        overlay = np.zeros((self.size, self.size, 2))
        overlay[:, :, 0] = detect_train_data
        overlay[:, :, 1] = clas_train_data
        overlay_single = overlay.max(axis=2)
        cv2.imwrite("resampled_overlay.png", overlay_single * 50)
        np.save('train_data.npy', overlay_single)


    def decide_clas_value(self, clas_value, nearest_detect_value):
        """
        convert clas_value to EF scale according to its own value and nearest_detect_value value
        :param clas_value:
        :param nearest_detect_value:
        :return: EF scale
        """
        # detection: 0(no data), 1(no damage), 2(EF0), 3(EF1), 4(EF2), 5(EF3)
        # classification: 0(no data), 1(no damage), 2(light damage), 3(severe damage)
        # nearest_detect_value in [1,5]
        if clas_value == 1:
            if nearest_detect_value < 2:
                return (1 + nearest_detect_value)/2.0
            else:
                return 1.5
        elif clas_value == 2:
            if nearest_detect_value < 1.5:
                return 1.5
            elif nearest_detect_value < 2.5:
                return (2 + nearest_detect_value)/2.0
            elif nearest_detect_value < 4:
                return (3 + nearest_detect_value)/2.0
            else:
                return 3.5
        elif clas_value == 3:
            if nearest_detect_value < 3:
                return 3.5
            elif nearest_detect_value < 4.5:
                return (4 + nearest_detect_value)/2.0
            else:
                return (5 + nearest_detect_value)/2.0
        else:
            print('error!!')

    def read_sample(self, sample_file):
        pass

    def recover_image(self, save_file):
        pass


if __name__ == '__main__':
    sp = SP()
    detect_file = 'resized_img_detect.png'
    clas_file = 'resized_img_clas.png'
    mask_file = 'resized_mask.png'
    sp.readOriginal(detect_file, clas_file, mask_file)
    grid = sp.grid_images(200)
    #sp.sample()