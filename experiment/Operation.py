import element_detection.detect_compo.lib_ip.ip_preprocessing as pre
import element_detection.detect_text.text_detection as text

import cv2
import numpy as np


class Operation:
    def __init__(self, ui_img_path, action, target_element_bounds):
        self.ui_img_path = ui_img_path
        self.ui_img = cv2.imread(ui_img_path)
        self.ui_img_height, self.ui_img_width = self.ui_img.shape[:2]

        self.action = action
        self.target_element_bounds = target_element_bounds  # [[left, top],[right, bottom]]
        self.target_element_img = None
        self.target_element_text = None

        self.clip_target_element_img()

    def clip_target_element_img(self):
        self.target_element_img = self.ui_img[self.target_element_bounds[0][1]: self.target_element_bounds[1][1], self.target_element_bounds[0][0]: self.target_element_bounds[1][0]]

    def resize(self, width_resize, height_resize):
        width_resize_ratio = width_resize / self.ui_img_width
        height_resize_ratio = height_resize / self.ui_img_height
        self.target_element_bounds[0][0] = int(self.target_element_bounds[0][0] * width_resize_ratio)
        self.target_element_bounds[0][1] = int(self.target_element_bounds[0][1] * height_resize_ratio)
        self.target_element_bounds[1][0] = int(self.target_element_bounds[1][0] * width_resize_ratio)
        self.target_element_bounds[1][1] = int(self.target_element_bounds[1][1] * height_resize_ratio)

        self.ui_img = cv2.resize(self.ui_img, (width_resize, height_resize))
        self.clip_target_element_img()

    def shrink_target_element_img(self, min_grad=6):
        '''
        :param min_grad: binarization threshold
        :param filled_ratio: the minimum ratio of a row/column to be filled
        '''
        img_binary = pre.binarization(self.target_element_img, min_grad)
        h, w = img_binary.shape
        img_one = img_binary // 255

        # left to right
        shrink_left = 0
        for i in range(0, w):
            if i <= 5:
                if np.sum(img_one[:, i]) > h * 0.8:
                    shrink_left = i
                    break
            else:
                if np.sum(img_one[:, i]) > 5:
                    shrink_left = i
                    break
        # right to left
        shrink_right = w
        for i in range(w - 1, 0, -1):
            if i >= w - 5:
                if np.sum(img_one[:, i]) > h * 0.8:
                    shrink_right = i
                    break
            else:
                if np.sum(img_one[:, i]) > 5:
                    shrink_right = i
                    break
        shrink_right = w - shrink_right

        # top down
        shrink_top = 0
        for i in range(0, h):
            if i <= 5:
                if np.sum(img_one[i, :]) > w * 0.8:
                    shrink_top = i
                    break
            else:
                if np.sum(img_one[i, :]) > 5:
                    shrink_top = i
                    break
        # bottom up
        shrink_bottom = h
        for i in range(h - 1, 0, -1):
            if i >= h - 5:
                if np.sum(img_one[i, :]) > w * 0.8:
                    shrink_bottom = i
                    break
            else:
                if np.sum(img_one[i, :]) > 5:
                    shrink_bottom = i
                    break
        shrink_bottom = h - shrink_bottom

        self.target_element_bounds[0][0] += shrink_left
        self.target_element_bounds[0][1] += shrink_top
        self.target_element_bounds[1][0] -= shrink_right
        self.target_element_bounds[1][1] -= shrink_bottom
        self.target_element_img = self.ui_img[self.target_element_bounds[0][1]: self.target_element_bounds[1][1], self.target_element_bounds[0][0]: self.target_element_bounds[1][0]]

        # cv2.imshow('bin', img_binary)
        # cv2.imshow('e', self.target_element_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    def detect_text(self, paddle_ocr):
        cv2.imwrite('data/operation/target.png', self.target_element_img)
        _, self.target_element_text = text.text_detection_paddle('data/operation/target.png', 'data/operation', paddle_cor=paddle_ocr)

    def get_text_content(self):
        return [t.content for t in self.target_element_text]

    def show_target_ele(self, show=True):
        board = self.ui_img.copy()
        cv2.rectangle(board, self.target_element_bounds[0], self.target_element_bounds[1], (255,0,0), 2)
        if show:
            cv2.imshow('target element', board)
            # cv2.imshow('clip', self.target_element_img)
            cv2.waitKey()
            # cv2.destroyAllWindows()
            cv2.destroyWindow('target element')
        return board
