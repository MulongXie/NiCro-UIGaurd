import cv2
from robot.robot_control import RobotController
from utils.GUI import GUI
from utils.UIGuard import UIGuard


class Robot(RobotController):
    def __init__(self, speed=100000, press_depth=20, dp_model_loader=None):
        super().__init__(speed=speed)
        self.id = -1
        self.press_depth = press_depth
        self.name = 'robot'

        self.camera = None  # height/width = 1000/540
        self.camera_clip_range_height = [111, 914]
        self.camera_clip_range_width = [0, 540]

        self.x_robot2y_cam = round((300-120)/(self.camera_clip_range_height[1] - self.camera_clip_range_height[0]), 2)    # x_robot_range : cam.height_range
        self.y_robot2x_cam = round(120/(self.camera_clip_range_width[1] - self.camera_clip_range_width[0]), 2)          # y_robot_range : cam.width_range

        self.GUI = None
        self.photo = None   # image
        self.photo_save_path = 'data/screen/robot_photo.png'
        # self.photo_screen_area = None    # image of screen area
        self.detect_resize_ratio = None  # self.GUI.detection_resize_height / self.photo.shape[0]
        self.dp_model_loader = dp_model_loader      # model loader for dark pattern detection
        self.cap_frame()

    def get_devices_info(self):
        print("Device Robot")

    def cap_frame(self):
        if not self.camera or not self.camera.read()[0]:
            self.camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        ret, frame = self.camera.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = frame[self.camera_clip_range_height[0]: self.camera_clip_range_height[1], self.camera_clip_range_width[0]:self.camera_clip_range_width[1]]
        self.photo = frame
        cv2.imwrite(self.photo_save_path, frame)
        return frame

    def convert_coord_from_camera_to_robot(self, x_cam, y_cam):
        x_robot = int(((self.camera_clip_range_height[1] - self.camera_clip_range_height[0]) - y_cam) * self.x_robot2y_cam) + 120
        y_robot = int(((self.camera_clip_range_width[1] - self.camera_clip_range_width[0]) / 2 - x_cam) * self.y_robot2x_cam)
        return x_robot, y_robot

    def adjust_camera_clip_range(self):
        def nothing(x):
            pass
        cv2.namedWindow('win')
        cv2.createTrackbar('top', 'win', self.camera_clip_range_height[0], 1000, nothing)
        cv2.createTrackbar('left', 'win', self.camera_clip_range_width[0], 540, nothing)
        cv2.createTrackbar('bottom', 'win', self.camera_clip_range_height[1], 1001, nothing)
        cv2.createTrackbar('right', 'win', self.camera_clip_range_width[1], 541, nothing)
        while 1:
            top = cv2.getTrackbarPos('top', 'win')
            left = cv2.getTrackbarPos('left', 'win')
            bottom = cv2.getTrackbarPos('bottom', 'win')
            right = cv2.getTrackbarPos('right', 'win')
            frame = self.cap_frame()
            frame_clip = frame[top:bottom, left:right]
            cv2.imshow('frame', frame)
            cv2.imshow('clip', frame_clip)
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyAllWindows()
        self.camera.release()

    def control_robot_by_clicking_on_cam_video(self):
        def click_event(event, x, y, flags, params):
            x_pre, y_pre = params
            if event == cv2.EVENT_LBUTTONDOWN:
                params[0], params[1] = self.convert_coord_from_camera_to_robot(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                x, y = self.convert_coord_from_camera_to_robot(x, y)
                print(x, y, params)
                # swipe
                if abs(x_pre - x) >= 10 or abs(y_pre - y) >= 10:
                    self.swipe((x_pre, y_pre, self.press_depth), (x, y, self.press_depth))
                # click
                else:
                    self.click((x_pre, y_pre, self.press_depth))
            elif event == cv2.EVENT_RBUTTONDOWN:
                x, y = self.convert_coord_from_camera_to_robot(x, y)
                self.shake((x, y, self.press_depth))
            elif event == cv2.EVENT_MOUSEWHEEL:
                x, y = self.convert_coord_from_camera_to_robot(x, y)
                self.push((x, y, self.press_depth))
        button_down_coords = [-1, -1]
        blur = False
        while 1:
            frame = self.cap_frame()
            if blur:
                frame = cv2.GaussianBlur(frame,(5,5),cv2.BORDER_DEFAULT)
            # get the click point on the image
            cv2.imshow('camera', frame)
            cv2.setMouseCallback('camera', click_event, param=button_down_coords)
            if cv2.waitKey(1) == ord('q'):
                break
            elif cv2.waitKey(1) == ord('b'):
                blur = not blur
        cv2.destroyWindow('camera')
        self.camera.release()

    def screen_swipe(self, x_start_screen, y_start_screen, x_end_screen, y_end_screen):
        x1_robot, y1_robot = self.convert_coord_from_camera_to_robot(x_start_screen, y_start_screen)
        x2_robot, y2_robot = self.convert_coord_from_camera_to_robot(x_end_screen, y_end_screen)
        self.swipe((x1_robot, y1_robot, self.press_depth), (x2_robot, y2_robot, self.press_depth))

    def screen_click(self, x_screen, y_screen):
        x_robot, y_robot = self.convert_coord_from_camera_to_robot(x_screen, y_screen)
        self.click((x_robot, y_robot, self.press_depth))

    def detect_dark_pattern(self, model_loader):
        '''
        Detect if the gui has dark pattern according to the GUI information
        '''
        self.GUI.classify_compos(model_loader=model_loader)
        dark_pattern = UIGuard(model_loader=model_loader)
        dark_pattern.detect_dark_pattern(image_path=self.GUI.img_path, elements_info=self.GUI.get_elements_info_ui_guard(), vis=False)

    def update_screenshot_and_gui(self, paddle_ocr, is_load=False, show=False, ocr_opt='paddle', adjust_by_screen_area=False, verbose=True, dp=False):
        self.cap_frame()
        self.GUI = GUI(self.photo_save_path)
        self.detect_resize_ratio = self.GUI.detection_resize_height / self.photo.shape[0]
        if is_load:
            self.GUI.load_detection_result()
        else:
            self.GUI.detect_element(True, True, True, paddle_ocr=paddle_ocr, ocr_opt=ocr_opt, verbose=verbose)
            self.detect_dark_pattern(self.dp_model_loader)
        if adjust_by_screen_area:
            self.adjust_elements_by_screen_area(show)
        elif show:
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

    '''
    **************************************
    *** Adjust Element by Phone Screen ***
    **************************************
    '''
    def recognize_phone_screen(self):
        gui = self.GUI
        for e in gui.elements:
            if e.height / gui.detection_resize_height > 0.5:
                if e.parent is None and e.children is not None:
                    e.is_screen = True
                    gui.screen = e
                    gui.img = e.clip
                    self.photo_screen_area = e.clip
                    return

    def remove_ele_out_screen(self):
        gui = self.GUI
        new_elements = []
        gui.ele_compos = []
        gui.ele_texts = []
        for ele in gui.elements:
            if ele.id in gui.screen.children:
                new_elements.append(ele)
                if ele.category == 'Compo':
                    gui.ele_compos.append(ele)
                elif ele.category == 'Text':
                    gui.ele_texts.append(ele)
        gui.elements = new_elements

    def convert_element_relative_pos_by_screen(self):
        gui = self.GUI
        s_left, s_top = gui.screen.col_min, gui.screen.row_min
        for ele in gui.elements:
            ele.col_min -= s_left
            ele.col_max -= s_left
            ele.row_min -= s_top
            ele.row_max -= s_top

    def resize_screen_and_elements_by_height(self):
        gui = self.GUI
        h_ratio = gui.detection_resize_height / gui.screen.height
        gui.screen.resize_bound(resize_ratio_col=h_ratio, resize_ratio_row=h_ratio)
        gui.img = cv2.resize(gui.img, (int(gui.screen.width * h_ratio), gui.detection_resize_height))
        for ele in gui.elements:
            ele.resize_bound(resize_ratio_col=h_ratio, resize_ratio_row=h_ratio)
            ele.get_clip(gui.img)
        gui.draw_detection_result()

    def adjust_elements_by_screen_area(self, show=False):
        '''
        Recognize the phone screen region if any and adjust the element coordinates according to the screen
        '''
        self.recognize_phone_screen()
        if self.GUI.screen is None:
            return
        self.remove_ele_out_screen()
        self.convert_element_relative_pos_by_screen()
        self.resize_screen_and_elements_by_height()
        if show:
            self.GUI.show_detection_result()

    def convert_element_pos_back(self, element):
        '''
        Convert back the element coordinates from the phone screen-based to the whole image-based (detection_resize_height)
        '''
        gui = self.GUI
        if gui.screen is None:
            return
        h_ratio = gui.screen.height / gui.detection_resize_height
        element.col_min = int((element.col_min + gui.screen.col_min) * h_ratio)
        element.col_max = int((element.col_max + gui.screen.col_min) * h_ratio)
        element.row_min = int((element.row_min + gui.screen.row_min) * h_ratio)
        element.row_max = int((element.row_max + gui.screen.row_min) * h_ratio)
        element.init_bound()
        element.get_clip(gui.img)

    def draw_elements_on_screen(self, show=True):
        gui = self.GUI
        board = gui.screen_img.copy()
        for ele in gui.elements:
            ele.draw_element(board, show=False)
        if show:
            cv2.imshow('Elements on screen', board)
            cv2.waitKey()
            cv2.destroyWindow('Elements on screen')
        return board

    '''
    *********************
    *** Action Replay ***
    *********************
    '''
    def replay_action(self, action, matched_element=None, screen_ratio=None, src_shape=None, phone_ratio_width=0.7):
        if action['type'] == 'click':
            if matched_element is not None:
                x_screen, y_screen = int(matched_element.center_x / self.detect_resize_ratio), int(matched_element.center_y / self.detect_resize_ratio)
                x_robot, y_robot = self.convert_coord_from_camera_to_robot(x_screen, y_screen)
                self.click((x_robot, y_robot, self.press_depth))
            else:
                x_screen, y_screen = int((action['coordinate'][0][0] / src_shape[0]) * self.photo.shape[1] * phone_ratio_width), int((action['coordinate'][0][1] / src_shape[1]) * self.photo.shape[0])
                x_robot, y_robot = self.convert_coord_from_camera_to_robot(x_screen, y_screen)
                print('Screen Coord(%d, %d), Robot Coord(%d, %d)' % (x_screen, y_screen, x_robot, y_robot), src_shape, self.photo.shape, action['coordinate'])
                self.click((x_robot, y_robot, self.press_depth))
        elif action['type'] == 'swipe':
            x_screen, y_screen = int((action['coordinate'][0][0] / src_shape[0]) * self.photo.shape[1] * phone_ratio_width), int((action['coordinate'][0][1] / src_shape[1]) * self.photo.shape[0])
            x_robot, y_robot = self.convert_coord_from_camera_to_robot(x_screen, y_screen)
            start_coord = (x_robot, y_robot, self.press_depth)
            x_screen_end, y_screen_end = int((action['coordinate'][1][0] / src_shape[0]) * self.photo.shape[1] * phone_ratio_width), int((action['coordinate'][1][1] / src_shape[1]) * self.photo.shape[0])
            x_robot_end, y_robot_end = self.convert_coord_from_camera_to_robot(x_screen_end, y_screen_end)
            end_coord = (x_robot_end, y_robot_end, self.press_depth)
            print('Swipe robot:', start_coord, end_coord, screen_ratio)
            self.swipe(start_coord, end_coord)


if __name__ == '__main__':
    robot = Robot(speed=1000000)
    robot.control_robot_by_clicking_on_cam_video()
