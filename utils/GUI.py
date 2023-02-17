import cv2
import json
import os
import numpy as np
from os.path import join as pjoin
from difflib import SequenceMatcher
import time
time.clock = time.time

from element_detection.Element import Element
from sklearn.metrics.pairwise import cosine_similarity
from element_detection.classify_compo.CNN import CNN


class GUI:
    def __init__(self, img_path='data/input', output_dir='data/output', detection_resize_height=800):
        self.img_path = img_path
        self.ui_name = img_path.replace('\\', '/').split('/')[-1].split('.')[0]
        self.output_dir = output_dir
        self.img_org = cv2.imread(self.img_path)

        self.detection_resize_height = detection_resize_height  # resize the input gui while detecting
        self.detection_resize_width = int(self.img_org.shape[1] * (self.detection_resize_height / self.img_org.shape[0]))
        self.img = cv2.resize(self.img_org, (self.detection_resize_width, self.detection_resize_height))

        self.det_result_imgs = {'text': None, 'non-text': None, 'merge': None}  # image visualization for different stages
        self.det_result_data = None         # {'compos':[], 'img_shape'}

        self.elements = []                  # list of Element objects for android UI
        self.ele_compos = []
        self.ele_texts = []
        self.elements_mapping = {}          # {'id': Element}
        self.has_popup_modal = False        # if the ui has popup modal
        self.selected_element = None        # the selected element by clicking

        self.screen = None      # Element object
        self.screen_img = None

    '''
    *******************************
    *** Detect or Load Elements ***
    *******************************
    '''
    def detect_element(self, is_text=True, is_nontext=True, is_merge=True, paddle_ocr=None, ocr_opt='paddle', verbose=True):
        '''
        :param is_text: Boolean, True to detect text
        :param is_nontext: Boolean, True to detect non-text
        :param is_merge: Boolean, True to merge text and non-text results
        :param ocr_opt: 'google' or 'paddle'
        :param verbose: Boolean, if output the processing time
        '''
        if is_text:
            os.makedirs(pjoin(self.output_dir, 'ocr'), exist_ok=True)
            import element_detection.detect_text.text_detection as text
            if ocr_opt == 'paddle':
                self.det_result_imgs['text'], _ = text.text_detection_paddle(self.img_path, pjoin(self.output_dir, 'ocr'), paddle_ocr=paddle_ocr, verbose=verbose)
            elif ocr_opt == 'google':
                self.det_result_imgs['text'], _ = text.text_detection_google(self.img_path, pjoin(self.output_dir, 'ocr'), verbose=verbose)
        if is_nontext:
            os.makedirs(pjoin(self.output_dir, 'ip'), exist_ok=True)
            import element_detection.detect_compo.ip_region_proposal as ip
            key_params = {'min-grad': 6, 'ffl-block': 5, 'min-ele-area': 50, 'merge-contained-ele': False}
            self.det_result_imgs['non-text'] = ip.compo_detection(self.img_path, self.output_dir, key_params, resize_by_height=self.detection_resize_height, adaptive_binarization=False, verbose=verbose)
        if is_merge:
            os.makedirs(pjoin(self.output_dir, 'merge'), exist_ok=True)
            import element_detection.detect_merge.merge as merge
            compo_path = pjoin(self.output_dir, 'ip', str(self.ui_name) + '.json')
            ocr_path = pjoin(self.output_dir, 'ocr', str(self.ui_name) + '.json')
            self.det_result_imgs['merge'], self.det_result_data = merge.merge(self.img_path, compo_path, ocr_path, pjoin(self.output_dir, 'merge'), is_remove_bar=True, is_paragraph=False, verbose=verbose)
            # convert elements as Element objects
            self.cvt_elements()

    def load_detection_result(self, data_path=None):
        if not data_path:
            data_path = pjoin(self.output_dir, 'merge', self.ui_name + '.json')
        self.det_result_data = json.load(open(data_path))
        # convert elements as Element objects
        self.cvt_elements()

    def classify_compos(self, classifier):
        '''
        Classify compos: ['Text Button', 'Input', 'Switch', 'Image', 'Icon', 'Checkbox']
        '''
        os.makedirs(pjoin(self.output_dir, 'cls'), exist_ok=True)
        save_file = pjoin(self.output_dir, 'cls', str(self.ui_name) + '.json')

        labels = classifier.predict_images([compo.clip for compo in self.ele_compos])
        result = {'compos': [], 'img_shape': self.img.shape}
        for i, compo in enumerate(self.ele_compos):
            compo.compo_class = labels[i]
            result['compos'].append(compo.wrap_info())
        for text in self.ele_texts:
            result['compos'].append(text.wrap_info())
        json.dump(result, open(save_file, 'w'), indent=4)

    '''
    **************************************
    *** Operations for Element Objects ***
    **************************************
    '''
    def cvt_elements(self):
        '''
        Convert detection result to Element objects
        @ det_result_data: {'elements':[], 'img_shape'}
        '''
        self.elements = []
        self.elements_mapping = {}
        for i, element in enumerate(self.det_result_data['compos']):
            e = Element(i, element['class'], element['position'])
            if element['class'] == 'Text':
                e.text_content = element['text_content']
                e.keyboard = element['keyboard']
            if 'children' in element:
                e.children = element['children']
            if 'parent' in element:
                e.parent = element['parent']
            e.get_clip(self.img)
            self.elements.append(e)
            self.elements_mapping[e.id] = e
        self.group_elements()

    def group_elements(self):
        self.ele_compos = []
        self.ele_texts = []
        for ele in self.elements:
            if ele.category == 'Compo':
                self.ele_compos.append(ele)
            elif ele.category == 'Text':
                self.ele_texts.append(ele)

    def save_element_clips(self):
        clip_dir = pjoin(self.output_dir, 'clip')
        os.makedirs(clip_dir, exist_ok=True)

        for element in self.elements:
            name = pjoin(clip_dir, element.id + '.jpg')
            cv2.imwrite(name, element.clip)

    '''
    **********************
    *** GUI Operations ***
    **********************
    '''
    def recognize_popup_modal(self, height_thresh=0.15, width_thresh=0.5):
        def is_element_modal(element, area_resize):
            gray = cv2.cvtColor(element.clip, cv2.COLOR_BGR2GRAY)
            area_ele = element.clip.shape[0] * element.clip.shape[1]
            # calc the grayscale of the element
            sum_gray_ele = np.sum(gray)
            mean_gray_ele = sum_gray_ele / area_ele
            # calc the grayscale of other region except the element
            sum_gray_other = sum_gray_a - sum_gray_ele
            mean_gray_other = sum_gray_other / (area_resize - area_ele)
            # if the element's brightness is far higher than other regions, it should be a pop-up modal
            if mean_gray_ele > 180 and mean_gray_other < 80:
                return True
            return False

        # calculate the mean pixel value as the brightness
        img_resized = self.img.copy()
        area_resize = img_resized.shape[0] * img_resized.shape[1]

        sum_gray_a = np.sum(cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY))

        if sum_gray_a / (img_resized.shape[0] * img_resized.shape[1]) < 100:
            for ele in self.elements:
                if ele.category == 'Compo' and \
                        ele.height / ele.detection_img_size[0] > height_thresh and ele.width / ele.detection_img_size[1] > width_thresh:
                    ele.get_clip(img_resized)
                    if is_element_modal(ele, area_resize):
                        self.has_popup_modal = True
                        ele.is_popup_modal = True
        if not self.has_popup_modal:
            print("No popup modal")

    def get_element_by_coordinate(self, x, y):
        '''
        Given a coordinate (x,y), get the leaf element in that position
        '''
        element = None
        for ele in self.elements:
            if ele.col_min <= x <= ele.col_max and ele.row_min <= y <= ele.row_max:
                if ele.children is None:
                    return ele
                else:
                    element = ele
        return element

    def get_element_by_clicking(self):
        '''
        Get the element by mouse clicking on the image
        '''

        def click_event(event, c_x, c_y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                x, y = c_x, c_y
                print('Click (%d, %d)' % (x, y))
                clicked_ele = self.get_element_by_coordinate(x, y)
                if clicked_ele is not None:
                    cv2.imshow('Clicked element', clicked_ele.clip)
                    self.selected_element = clicked_ele
                else:
                    print('No element in the click position')

        cv2.imshow('detection result', self.det_result_imgs['merge'])
        cv2.setMouseCallback('detection result', click_event)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return self.selected_element

    '''
    *************************
    *** Elements Matching ***
    *************************
    '''
    def match_element_template_matching(self, target_ele_img, show=False):
        if self.screen_img is not None:
            img_gray = cv2.cvtColor(self.screen_img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        target_img_gray = cv2.cvtColor(target_ele_img, cv2.COLOR_BGR2GRAY)
        w, h = target_img_gray.shape[::-1]

        # methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(img_gray, target_img_gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        element = Element(element_id=len(self.elements), category='Compo', position={'column_min':top_left[0], 'row_min':top_left[1], 'column_max':bottom_right[0], 'row_max':bottom_right[1]})
        if show:
            board = self.img.copy()
            element.draw_element(board, show=True)
        return element

    '''
    *********************
    *** Visualization ***
    *********************
    '''
    def show_detection_result(self):
        if self.det_result_imgs['merge'] is not None:
            cv2.imshow('GUI detection result (Press any key to exit)', self.det_result_imgs['merge'])
        elif self.det_result_data is not None:
            self.draw_detection_result()
            cv2.imshow('GUI detection result (Press any key to exit)', self.det_result_imgs['merge'])
        else:
            print('No detection result, run element_detection() or load_detection_result() first')
        cv2.waitKey()
        cv2.destroyWindow('GUI detection result (Press any key to exit)')

    def show_compo_cls_result(self):
        color_map = {'Text Button':(0,0,255), 'Input':(166,0,0), 'Switch':(166,166,0), 'Image':(0,166,166), 'Icon':(255,255,0), 'Checkbox':(255,0,166)}
        board = self.img.copy()
        for compo in self.ele_compos:
            compo.draw_element(board, color_map[compo.compo_class], put_text=compo.compo_class)
        cv2.imshow('Compo Classification Result', board)
        cv2.waitKey()
        cv2.destroyWindow('Compo Classification Result')

    def draw_detection_result(self):
        '''
        Draw detected elements based on det_result_data
        '''
        color_map = {'Compo': (0,255,0), 'Text': (0,0,255), 'Block':(0,255,255)}

        board = self.img.copy()
        for i, element in enumerate(self.elements):
            element.draw_element(board, color_map[element.category])
        self.det_result_imgs['merge'] = board.copy()
        return self.det_result_imgs['merge']

    def draw_popup_modal(self):
        if self.has_popup_modal:
            board = self.img.copy()
            for ele in self.elements:
                if ele.is_popup_modal:
                    ele.draw_element(board, color=(0,0,255), line=5, show_id=False)
            cv2.putText(board, 'popup modal', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
            cv2.imshow('modal', board)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def draw_screen(self, show=True, extract=True):
        board = self.img.copy()
        if self.screen is not None:
            self.screen.draw_element(board, color=(255,0,255), line=5, show_id=False)
            if extract:
                board = self.screen.clip
        if show:
            cv2.imshow('screen', board)
            cv2.waitKey()
            cv2.destroyAllWindows()
        return board

    def draw_element_after_screen_conversion(self, show=True):
        '''
        draw all the elements after adjust_elements_by_screen;
        '''
        if self.screen is None:
            print('No screen conversion. Call adjust_elements_by_screen() first')
            return
        board = self.screen_img.copy()
        for ele in self.elements:
            ele.draw_element(board, show=False, show_id=False)
        if show:
            cv2.imshow('Screen elements', board)
            cv2.waitKey()
            cv2.destroyAllWindows()
        return board

    def get_elements_info_ui_guard(self):
        ele_info = []
        for ele in self.elements:
            ele_info.append(ele.wrap_info_ui_guard())
        return ele_info


if __name__ == '__main__':
    gui = GUI('../data/input/2.png')
    gui.detect_element(True, True, True, ocr_opt='google')
    gui.show_detection_result()

    cnn = CNN()
    cnn.load()
    gui.classify_compos(cnn)
    gui.show_compo_cls_result()
