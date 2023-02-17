import time
time.clock = time.time

import cv2
import numpy as np
import os
from os.path import join as pjoin

from utils.Device import Device
from element_matching.GUI_pair import GUIPair
from utils.Robot import Robot

from ppadb.client import Client as AdbClient
client = AdbClient(host="127.0.0.1", port=5037)

# from paddleocr import PaddleOCR
# paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

from keras.applications.resnet import ResNet50
resnet_model = ResNet50(include_top=False, input_shape=(32, 32, 3))


class NiCro:
    def __init__(self, ocr_opt='google'):
        # Device objects, including their screenshots and GUIs
        self.devices = []            # list of Device
        self.source_device = None    # the selected source Device

        # the action on the GUI
        # 'type': click, swipe, long press
        # 'coordinate': action target coordinates in device screen size: 'click' has one coord, 'swipe' has two [start, end]
        self.action = {'type': 'click', 'coordinate': [(-1, -1), (-1, -1)]}
        self.target_element = None

        self.ocr_opt = ocr_opt             # 'paddle' or 'google'
        # self.paddle_ocr = paddle_ocr
        self.paddle_ocr = None
        self.resnet_model = resnet_model   # resnet encoder for image matching

        self.robot = None

        self.widget_matching_acc = {'sift':[], 'resnet':[], 'template-match':[], 'text':[], 'nicro':[]}
        self.test_round = 0

    '''
    *********************************
    *** Device & Robot Connection ***
    *********************************
    '''
    def load_devices(self):
        self.devices = [Device(i, dev) for i, dev in enumerate(sorted(client.devices(), key=lambda x: x.get_serial_no()))]
        self.source_device = self.devices[0]
        print('Load %d Device Emulators' % len(self.devices), 'Set Device 0 as the Source Device')

    def load_robot(self):
        self.robot = Robot()
        print('Robot Arm System Ready')

    def get_devices_info(self):
        print('Selected Source Device:')
        self.source_device.get_devices_info()
        print('\nDevice Emulators:')
        for i, dev in enumerate(self.devices):
            dev.get_devices_info()
        if self.robot is not None:
            print('Robot Arm System Loaded')
        else:
            print('No Robot Arm System')

    def select_source_device(self, device_id):
        self.source_device = self.devices[device_id]
        self.get_devices_info()

    def detect_gui_info_for_all_devices(self, load_detection_result=False, show=True, verbose=True, dark_pattern=False, model_loader=None):
        '''
        Detect GUI elements from the GUI images of all devices
        :param load_detection_result: True to load previous detection result if any
        :param show: True to visualize the detection results
        :param verbose: True to print out the detailed log of detection
        :param dark_pattern: True to detect dark pattern
        :param model_loader: ModelLoader for dark pattern detection
        '''
        for i, device in enumerate(self.devices):
            print('\n****** GUI Component Detection Device [%d / %d] ******' % (i + 1, len(self.devices)))
            device.update_screenshot_and_gui(self.paddle_ocr, load_detection_result, show, ocr_opt=self.ocr_opt, verbose=verbose)
            if dark_pattern:
                print('*** Dark Pattern Detection ***')
                device.detect_dark_pattern(model_loader)
        if self.robot is not None:
            print('\n****** GUI Component Detection Robot Arm [1 / 1] ******')
            self.robot.detect_gui_element(self.paddle_ocr, load_detection_result, show=show, ocr_opt=self.ocr_opt, verbose=verbose)
            if dark_pattern:
                print('*** Dark Pattern Detection ***')
                self.robot.detect_dark_pattern(model_loader)

    def show_all_device_detection_results(self):
        for device in self.devices:
            device.GUI.show_detection_result()

    '''
    *************************
    *** Cross Dev Actions ***
    *************************
    '''
    def record_and_replay_actions(self, output_root, app_name, testcase_id,
                                  is_record=True, is_replay=False):
        '''
        Perform cross-device record or replay
        :param output_root: The root directory to store actions
        :param app_name: The name of the app under test
        :param testcase_id: The id of the current testcase for the app
        :param is_record: Boolean, True to store the actions on the recording device
        :param is_replay: Boolean, True to replay and store the actions on all devices
        '''
        s_dev = self.source_device
        win_name = s_dev.device.get_serial_no() + ' screen (Press "q" to exit)'

        def on_mouse(event, x, y, flags, params):
            '''
            :param params: [board (image), step number (int), is pressing (boolean), press time (float)]
            :param x, y: in the scale of detection image size (height=800)
            '''
            x_app, y_app = int(x / s_dev.detect_resize_ratio), int(y / s_dev.detect_resize_ratio)
            # Press button
            if event == cv2.EVENT_LBUTTONDOWN:
                params[2] = True
                # draw the press location
                cv2.circle(params[0], (x, y), 10, (255, 0, 255), 2)
                cv2.imshow(win_name, params[0])
                self.action['coordinate'][0] = (x_app, y_app)
                # reset press_time
                params[3] = time.time()
            # Drag
            elif params[2] and event == cv2.EVENT_MOUSEMOVE:
                cv2.circle(params[0], (x, y), 10, (255, 0, 255), 2)
                cv2.imshow(win_name, params[0])
            # Lift button
            elif event == cv2.EVENT_LBUTTONUP:
                params[2] = False
                x_start, y_start = self.action['coordinate'][0]
                # swipe
                if abs(x_start - x_app) >= 10 or abs(y_start - y_app) >= 10:
                    print('\n****** Scroll from (%d, %d) to (%d, %d) ******' % (x_start, y_start, x_app, y_app))
                    s_dev.device.input_swipe(x_start, y_start, x_app, y_app, 500)
                    # record action
                    self.action['type'] = 'swipe'
                    self.action['coordinate'][1] = (x_app, y_app)
                # long press
                elif time.time() - params[3] >= 1:
                    print('\n****** Long Press (%d, %d) ******' % (x_start, y_start))
                    s_dev.device.input_swipe(x_start, y_start, x_start, y_start, 1000)
                    self.action['type'] = 'long press'
                    self.action['coordinate'][1] = (-1, -1)
                    cv2.circle(params[0], (x, y), 10, (255, 0, 255), -1)
                    cv2.imshow(win_name, params[0])
                # click
                else:
                    print('\n****** Click (%d, %d) ******' % (x_start, y_start))
                    s_dev.device.input_tap(x_start, y_start)
                    # record action
                    self.action['type'] = 'click'
                    self.action['coordinate'][1] = (-1, -1)

                step_id = str(params[1])
                # save original GUI image and operations on detection result
                if is_record:
                    print('Record action %s to %s' % (step_id, testcase_dir))
                    cv2.imwrite(pjoin(testcase_dir, step_id + '_org.jpg'), s_dev.GUI.img)  # original GUI screenshot
                    cv2.imwrite(pjoin(testcase_dir, step_id + '_act.jpg'), params[0])      # actions drawn on detection result
                # replay the action on all devices
                if is_replay:
                    save_action_execution_dir = pjoin(testcase_dir, step_id)
                    os.makedirs(save_action_execution_dir, exist_ok=True)
                    cv2.imwrite(pjoin(save_action_execution_dir, str(s_dev.id) + '.jpg'), params[0])      # actions drawn on detection result
                    self.replay_action_on_all_devices(detection_verbose=False, save_action_execution_dir=save_action_execution_dir)

                # update the screenshot and GUI of the selected target device
                print("****** Re-detect Source Device's screenshot and GUI ******")
                s_dev.update_screenshot_and_gui(self.paddle_ocr, ocr_opt=self.ocr_opt, verbose=False)
                params[0] = s_dev.GUI.det_result_imgs['merge'].copy()
                cv2.imshow(win_name, params[0])
                params[1] += 1

        # initiate the data directory to store actions
        testcase_dir = pjoin(output_root, app_name, testcase_id)
        os.makedirs(testcase_dir, exist_ok=True)

        # parameters sent to on_mouse: [board (image), step number (int), is pressing (boolean), press time (float)]
        board = s_dev.GUI.det_result_imgs['merge'].copy()  # board to visualize the action
        step = 0  # step number of the action in a test case
        press_time = time.time()  # record the time of pressing down, for checking long press action

        cv2.imshow(win_name, board)
        cv2.setMouseCallback(win_name, on_mouse, [board, step, False, press_time])
        cv2.waitKey()
        cv2.destroyWindow(win_name)

    '''
    **********************
    *** Replay Actions ***
    **********************
    '''
    def replay_action_on_device(self, device, detection_verbose=True, save_action_execution_path=None):
        print('*** Replay Devices Number [%d/%d] ***' % (device.id + 1, len(self.devices)))
        device.get_devices_info()
        # if widget-independent, match target widget first
        matched_element = None
        if self.target_element is not None:
            matched_element = GUIPair(self.source_device.GUI, device.GUI, self.resnet_model).match_target_element(self.target_element)
            # scroll down and match again
            if matched_element is None:
                print('Scroll down and try to match again')
                device.device.input_swipe(50, device.device.wm_size().height * 0.5, 50, 20, 500)
                device.update_screenshot_and_gui(self.paddle_ocr, ocr_opt=self.ocr_opt, verbose=detection_verbose)
                matched_element = GUIPair(self.source_device.GUI, device.GUI, self.resnet_model).match_target_element(self.target_element)
        # replay action on device
        device.replay_action(self.action, self.source_device.device.wm_size(), matched_element, save_action_execution_path)

    def replay_action_on_robot(self):
        print('*** Replay on Robot ***')
        screen_area_actual_height = self.robot.photo_screen_area.shape[0] / self.robot.detect_resize_ratio
        screen_ratio = self.source_device.device.wm_size()[1] / screen_area_actual_height
        matched_element = None
        if self.target_element is not None:
            gui_matcher = GUIPair(self.source_device.GUI, self.robot.GUI, self.resnet_model)
            matched_element = gui_matcher.match_target_element(self.target_element)
        self.robot.replay_action(self.action, matched_element, screen_ratio)

    def replay_action_on_all_devices(self, detection_verbose=True, save_action_execution_dir=None):
        print('Action:', self.action)
        # check if its widget dependent or independent
        self.target_element = self.source_device.find_element_by_coordinate(self.action['coordinate'][0][0], self.action['coordinate'][0][1], show=False)
        for dev in self.devices:
            save_action_execution_path = pjoin(save_action_execution_dir, str(dev.id) + '.jpg')
            # skip the Selected Source Device
            if dev.id != self.source_device.id:
                self.replay_action_on_device(dev, detection_verbose, save_action_execution_path)
            dev.update_screenshot_and_gui(self.paddle_ocr, ocr_opt=self.ocr_opt, verbose=detection_verbose)
        if self.robot is not None:
            self.replay_action_on_robot()
            self.robot.detect_gui_element(self.paddle_ocr, ocr_opt=self.ocr_opt, verbose=detection_verbose)

    '''
    ****************************
    *** Test Widget Matching ***
    ****************************
    '''
    def reset_matching_accuracy(self):
        self.test_round = 0
        device_num = len(self.devices) + 1 if self.robot is not None else len(self.devices)
        for method in self.widget_matching_acc:
            self.widget_matching_acc[method] = np.zeros(device_num)

    def match_widgets_cross_device(self, method):
        '''
        Match the target widget on all devices using the given method
        Press "a" if the match is correct
        :param method: 'sift', 'orb', 'resnet', 'template-match', 'text', 'nicro'
        '''
        print('*** Matching Method: %s in Test Round %d ***' % (method, self.test_round))
        all_devices = self.devices + [self.robot] if self.robot is not None else self.devices
        for i, dev in enumerate(all_devices):
            if dev.name == self.source_device.name:
                continue
            gui_matcher = GUIPair(self.source_device.GUI, dev.GUI, self.resnet_model)
            if method == 'nicro':
                matched_element = gui_matcher.match_target_element(self.target_element, show=False)
            else:
                matched_element = gui_matcher.match_target_element_test(self.target_element, method=method, show=False)
            key = gui_matcher.show_target_and_matched_elements(self.target_element, [matched_element])
            # correct matching
            if key == ord('a'):
                self.widget_matching_acc[method][i] += 1
            print('Device:', dev.name, self.widget_matching_acc[method])

    def click_to_match_widgets_cross_devices(self, method='nicro'):
        '''
        Match the clicked elements on the source devices through target devices
        :param method: the matching method
            @ 'nicro': the comprehensive matching approach of NiCro
            @ 'sift', 'resnet', 'template-match', 'text'
            @ 'all': iterate all methods
        '''
        s_dev = self.source_device
        win_name = s_dev.device.get_serial_no() + ' screen'

        def on_mouse(event, x, y, flags, params):
            '''
            :param params: [board (image)]
            :param x, y: in the scale of detection image size (height=800)
            '''
            x_app, y_app = int(x / s_dev.detect_resize_ratio), int(y / s_dev.detect_resize_ratio)
            # Press button
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(params[0], (x, y), 10, (255,0,255), 2)
                cv2.imshow(win_name, params[0])
                self.action['coordinate'][0] = (x_app, y_app)
            # Lift button
            elif event == cv2.EVENT_LBUTTONUP:
                self.test_round += 1
                print('\n****** Test Round: %d ******' % self.test_round)
                self.target_element = self.source_device.find_element_by_coordinate(self.action['coordinate'][0][0], self.action['coordinate'][0][1], show=False)
                if params[1] == 'all':
                    methods = ['sift', 'resnet', 'template-match', 'text', 'nicro']
                else:
                    methods = [params[1]]
                for m in methods:
                    self.match_widgets_cross_device(m)
                print(self.widget_matching_acc)
                params[0] = s_dev.GUI.det_result_imgs['merge'].copy()
                cv2.imshow(win_name, params[0])

        board = s_dev.GUI.det_result_imgs['merge'].copy()
        cv2.imshow(win_name, board)
        cv2.setMouseCallback(win_name, on_mouse, [board, method])
        key = cv2.waitKey()
        if key == ord('q'):
            cv2.destroyWindow(win_name)
            return
        cv2.destroyWindow(win_name)


if __name__ == '__main__':
    # 0. initiate NiCro with ocr option (paddle or google)
    nicro = NiCro(ocr_opt='google')

    # 1. load virtual devices and select one of them as the source device
    nicro.load_devices()
    nicro.select_source_device(0)

    # 2. load robot system
    # nicro.load_robot()
    # nicro.robot.control_robot_by_clicking_on_cam_video()   # test the robot system
    # nicro.robot.detect_gui_element(nicro.paddle_ocr, show=True, ocr_opt=nicro.ocr_opt)   # detect GUI elements from photo (screen region)
    # nicro.robot.GUI.screen.show_clip()    # show the screen region

    # 3. detect GUI components for all the devices for their current GUI
    nicro.detect_gui_info_for_all_devices(verbose=False, show=True)

    # 4. control the source device by mouse and replay the action on all other devices
    nicro.record_and_replay_actions(output_root='/home/ml/Data/visual testing/testcase', app_name='Desktop', testcase_id='1',
                                    is_record=True,     # record or replay
                                    is_replay=False)

    # 5. test the widget matching among all devices
    # nicro.reset_matching_accuracy()
    # nicro.click_to_match_widgets_cross_devices()
    # print(nicro.widget_matching_acc)
