import cv2
import time
from utils.GUI import GUI
from utils.UIGuard import UIGuard


class Device:
    def __init__(self, dev_id, device, dp_model_loader=None):
        self.id = dev_id
        self.name = device.get_serial_no()
        self.screenshot_path = 'data/screen/' + str(self.name) + '.png'
        self.device = device                        # ppadb device

        self.screenshot = self.cap_screenshot()     # cv2.image
        self.GUI = GUI(self.screenshot_path)        # GUI object
        self.detect_resize_ratio = self.GUI.detection_resize_height / self.screenshot.shape[0]
        self.dp_model_loader = dp_model_loader      # model loader for dark pattern detection

        # the action on the GUI
        # 'type': click, swipe
        # 'coordinate': action target coordinates, click has one coord, swipe has two [start, end]
        self.action = {'type': None, 'coordinate': [(-1, -1), (-1, -1)]}

    def get_devices_info(self):
        print("Device ID:%d Name:%s Resolution:%s" % (self.id, self.name, self.device.wm_size()))

    def cap_screenshot(self, recur_time=0):
        screen = self.device.screencap()
        with open(self.screenshot_path, "wb") as fp:
            fp.write(screen)
        self.screenshot = cv2.imread(self.screenshot_path)
        if recur_time < 3 and self.screenshot is None:
            self.cap_screenshot(recur_time+1)
        return self.screenshot

    def detect_gui_info(self, paddle_ocr, is_load=False, show=False, ocr_opt='paddle', verbose=True):
        self.GUI = GUI(self.screenshot_path)
        if is_load:
            self.GUI.load_detection_result()
        else:
            self.GUI.detect_element(True, True, True, paddle_ocr=paddle_ocr, ocr_opt=ocr_opt, verbose=verbose)
        if show:
            self.GUI.show_detection_result()

    def detect_dark_pattern(self, model_loader):
        '''
        Detect if the gui has dark pattern according to the GUI information
        '''
        print('*** Dark Pattern Detection ***')
        self.GUI.classify_compos(model_loader=model_loader)
        dark_pattern = UIGuard(model_loader=model_loader)
        dark_pattern.detect_dark_pattern(image_path=self.GUI.img_path, elements_info=self.GUI.get_elements_info_ui_guard(), vis=False)

    def update_screenshot_and_gui(self, paddle_ocr, is_load=False, show=False, ocr_opt='paddle', verbose=True):
        self.cap_screenshot()
        self.detect_gui_info(paddle_ocr, is_load=is_load, ocr_opt=ocr_opt, verbose=verbose)
        self.detect_dark_pattern(self.dp_model_loader)
        if show:
            self.GUI.show_detection_result()

    def find_element_by_coordinate(self, x, y, show=False):
        '''
        x, y: in the scale of app screen size
        '''
        x_resize, y_resize = x * self.detect_resize_ratio, y * self.detect_resize_ratio
        ele = self.GUI.get_element_by_coordinate(x_resize, y_resize)
        if ele is None:
            print('No element found at (%d, %d)' % (x_resize, y_resize))
        elif show:
            ele.show_clip()
        return ele

    def replay_action(self, action, source_dev_size, matched_element=None, save_action_execution_path=None):
        '''
        :param action: {'type': 'click', 'coordinate': [(-1, -1), (-1, -1)]}
        :param source_dev_size: (screen width, screen height)
        :param matched_element: Element object, element matched in the target device
        :param save_action_execution_path: The path to save the visualized action execution
        '''
        dev_size = self.device.wm_size()
        dev_ratio_x = dev_size[0] / source_dev_size[0]
        dev_ratio_y = dev_size[1] / source_dev_size[1]
        # get the target action coordinates on the device
        if matched_element is not None:
            coord = (int(matched_element.center_x / self.detect_resize_ratio), int(matched_element.center_y / self.detect_resize_ratio))
        else:
            coord = (int(action['coordinate'][0][0] * dev_ratio_x), int(action['coordinate'][0][1] * dev_ratio_y))

        # click
        if action['type'] == 'click':
            self.execute_action('click', [coord, (-1, -1)], save_action_execution_path)
        # long press
        elif action['type'] == 'long press':
            self.execute_action('long press', [coord, coord], save_action_execution_path)
        # swipe
        elif action['type'] == 'swipe':
            dist_x = action['coordinate'][1][0] - action['coordinate'][0][0]
            dist_y = int((action['coordinate'][1][1] - action['coordinate'][0][1]) * dev_ratio_y)
            if matched_element is not None:
                dev_x1 = coord[0]
                dev_y1 = coord[1]
            else:
                dev_x1 = int(action['coordinate'][0][0] * dev_ratio_x)
                dev_y1 = int(action['coordinate'][0][1] * dev_ratio_y)
            self.execute_action('swipe', [(dev_x1, dev_y1), (dev_x1 + dist_x, dev_y1 + dist_y)], save_action_execution_path)

    def execute_action(self, action_type, coordinates, save_action_execution_path=None):
        if action_type == 'click':
            self.device.input_tap(coordinates[0][0], coordinates[0][1])
        elif action_type == 'long press':
            self.device.input_swipe(coordinates[0][0], coordinates[0][1], coordinates[0][0], coordinates[0][1], 1000)
        elif action_type == 'swipe':
            self.device.input_swipe(coordinates[0][0], coordinates[0][1], coordinates[1][0], coordinates[1][1], 300)
        if save_action_execution_path is not None:
            self.save_action_execution(action_type, coordinates, save_action_execution_path)

    def save_action_execution(self, action_type, coordinates, save_path, num_dots=5, show=True):
        coordinates = ((int(coordinates[0][0] * self.detect_resize_ratio), int(coordinates[0][1] * self.detect_resize_ratio)),
                       (int(coordinates[1][0] * self.detect_resize_ratio), int(coordinates[1][1] * self.detect_resize_ratio)))
        board = self.GUI.det_result_imgs['merge'].copy()
        coord1, coord2 = coordinates
        if action_type == 'click':
            cv2.circle(board, coord1, 10, (255, 0, 255), 2)
        elif action_type == 'long press':
            print('long press')
            cv2.circle(board, coord1, 10, (255, 0, 255), -1)
        elif action_type == 'swipe':
            x_gap = (coord2[0] - coord1[0]) // num_dots
            y_gap = (coord2[1] - coord1[1]) // num_dots
            for i in range(num_dots):
                cv2.circle(board, (coord1[0] + i * x_gap, coord1[1] + i * y_gap), 10, (255, 0, 255), 2)
        cv2.imwrite(save_path, board)
        if show:
            cv2.imshow('action (Press any key to exit)', board)
            cv2.waitKey()
            cv2.destroyWindow('action (Press any key to exit)')
