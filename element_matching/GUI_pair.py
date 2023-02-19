import cv2
import json
import os
import numpy as np
import time
import shutil
from os.path import join as pjoin
from random import randint as rint
from glob import glob
from difflib import SequenceMatcher

import element_matching.matching as matching
from sklearn.metrics.pairwise import cosine_similarity


class GUIPair:
    def __init__(self, gui1, gui2, resnet_model=None):
        self.gui1 = gui1
        self.gui2 = gui2

        self.min_similarity_text = 0.85
        self.min_similarity_img = 0.8
        self.min_shape_difference = 0.8  # min / max

        # the similarity matrix of all elements in gui1 and gui2, shape: (len(gui1.all_elements), len(gui2.all_elements)
        self.image_similarity_matrix = None
        # the preload resnet model for image encoding
        self.resnet_model = resnet_model

    '''
    ******************************
    *** Match Similar Elements ***
    ******************************
    '''
    def calculate_elements_image_similarity_matrix(self):
        # calculate the similarity matrix through resnet for all elements in gui1 and gui2
        clips1 = [ele.clip for ele in self.gui1.elements]
        clips2 = [ele.clip for ele in self.gui2.elements]
        self.image_similarity_matrix = matching.image_similarity_matrix(clips1, clips2, method='resnet', resnet_model=self.resnet_model)

    def match_by_text(self, target_element, compare_elements, show=True):
        target_ele_text = target_element.text_content
        if target_ele_text is None:
            return []
        if target_ele_text is not list:
            target_ele_text = [target_ele_text]
        else:
            target_ele_text = sorted(target_ele_text, key=lambda x: len(x), reverse=True)
        matched_elements = []
        similarities = []
        is_matched = False
        for tar in target_ele_text:
            for text_ele in compare_elements:
                if text_ele.text_content is None or text_ele.keyboard != target_element.keyboard:
                    continue
                sim = SequenceMatcher(None, text_ele.text_content.lower(), tar.lower()).ratio()
                if sim > self.min_similarity_text:
                    matched_elements.append(text_ele)
                    similarities.append(sim)
                    is_matched = True
            # only use the longest matched text
            if is_matched:
                break
        # sort by similarity
        sorted_id = np.argsort(similarities)[::-1]
        matched_elements = np.array(matched_elements)[sorted_id]
        if show:
            self.show_target_and_matched_elements(target_element, matched_elements, similarities=similarities)
        return matched_elements

    def match_by_img(self, target_element, compared_elements, hash_check=False, show=True):
        # similarities between the target element and all elements in gui2
        resnet_sims = matching.image_similarity_matrix([target_element.clip], [e.clip for e in compared_elements], method='resnet', resnet_model=self.resnet_model)[0]
        # filter by similarity threshold
        matched_elements_id = np.where(resnet_sims > self.min_similarity_img)[0]
        # sort by similarity
        sorted_id = np.argsort(resnet_sims[matched_elements_id])[::-1]  # get the index
        matched_elements_id = matched_elements_id[sorted_id]
        # select from the compared_elements
        matched_elements = np.array(compared_elements)[matched_elements_id]
        if show:
            self.show_target_and_matched_elements(target_element, matched_elements, similarities=resnet_sims[matched_elements_id])

        # double check by dhash
        if hash_check and len(matched_elements) > 0:
            dhash_sims = matching.image_similarity_matrix([target_element.clip], [e.clip for e in matched_elements], method='dhash')[0]
            matched_elements_id = np.where(dhash_sims > self.min_similarity_img)[0]
            sorted_id = np.argsort(dhash_sims[matched_elements_id])[::-1]  # get the index
            matched_elements_id = matched_elements_id[sorted_id]
            matched_elements = matched_elements[matched_elements_id]
            # self.show_target_and_matched_elements(target_element, matched_elements, similarities=dhash_sims[matched_elements_id])
        return matched_elements

    def match_by_shape(self, target_element, compared_elements):
        matched_elements = []
        for ele in compared_elements:
            if (min(target_element.aspect_ratio, ele.aspect_ratio) / max(target_element.aspect_ratio, ele.aspect_ratio)) > self.min_shape_difference:
                matched_elements.append(ele)
        return matched_elements

    def match_by_neighbour(self, target_element, compared_elements):
        matched_elements = []
        if self.image_similarity_matrix is None:
            self.calculate_elements_image_similarity_matrix()
        return matched_elements

    def match_target_element(self, target_element, show=False):
        if target_element.category == 'Text' or target_element.text_content is not None:
            matched_elements = self.match_by_text(target_element, self.gui2.ele_texts, show=False)
            if len(matched_elements) > 1:
                matched_elements = self.match_by_shape(target_element, matched_elements)
            if len(matched_elements) > 1:
                matched_elements = self.match_by_img(target_element, matched_elements)
            # if len(matched_elements) > 1:
            #     matched_elements = self.match_by_neighbour(target_element, matched_elements)
        else:
            matched_elements = self.match_by_shape(target_element, self.gui2.ele_compos)
            if len(matched_elements) > 1:
                matched_elements = self.match_by_img(target_element, matched_elements, show=False)
            # if len(matched_elements) > 1:
            #     matched_elements = self.match_by_neighbour(target_element, matched_elements)
        if len(matched_elements) > 0:
            print('Successfully match element')
            if show:
                self.show_target_and_matched_elements(target_element, [matched_elements[0]])
            return matched_elements[0]
        else:
            print('No element matched')
            return None

    def match_target_element_test(self, target_element, method='sift', show=False):
        if target_element.category == 'Text':
            compared_elements = self.gui2.ele_texts
        else:
            compared_elements = self.gui2.ele_compos
        matched_element = None
        max_sim = None
        if method == 'sift':
            similarities = matching.image_similarity_matrix([target_element.clip], [e.clip for e in compared_elements], method='sift')[0]
            matched_element = compared_elements[np.argmax(similarities)]
            max_sim = [np.max(similarities)]
        elif method == 'surf':
            similarities = matching.image_similarity_matrix([target_element.clip], [e.clip for e in compared_elements], method='surf')[0]
            matched_element = compared_elements[np.argmax(similarities)]
            max_sim = [np.max(similarities)]
        elif method == 'orb':
            similarities = matching.image_similarity_matrix([target_element.clip], [e.clip for e in compared_elements], method='orb')[0]
            matched_element = compared_elements[np.argmax(similarities)]
            max_sim = [np.max(similarities)]
        elif method == 'resnet':
            similarities = matching.image_similarity_matrix([target_element.clip], [e.clip for e in compared_elements], method='resnet', resnet_model=self.resnet_model)[0]
            matched_element = compared_elements[np.argmax(similarities)]
            max_sim = [np.max(similarities)]
        elif method == 'template-match':
            matched_element = matching.match_element_template_matching(self.gui2.img, target_element.clip)
        elif method == 'text':
            matched_element = self.match_by_text(target_element, compared_elements)
            if len(matched_element) == 0: matched_element = None
            else: matched_element = matched_element[0]

        if show:
            if matched_element is not None:
                self.show_target_and_matched_elements(target_element, [matched_element], max_sim)
            else:
                self.show_target_and_matched_elements(target_element, [], None)
        return matched_element

    '''
    *********************
    *** Visualization ***
    *********************
    '''
    def show_detection_result(self):
        rest1 = self.gui1.draw_detection_result()
        rest2 = self.gui2.draw_detection_result()
        cv2.imshow('detection1', rest1)
        cv2.imshow('detection2', rest2)
        cv2.waitKey()
        cv2.destroyWindow('detection1')
        cv2.destroyWindow('detection2')

    def show_target_and_matched_elements(self, target, matched_elements, match_result_save_path=None, similarities=None):
        board1 = self.gui1.det_result_imgs['merge'].copy()
        board2 = self.gui2.det_result_imgs['merge'].copy()
        target.draw_element(board1, show=False)
        for i, ele in enumerate(matched_elements):
            if ele is None:
                continue
            if similarities is not None:
                text = similarities[i]
                cv2.putText(board2, text, (ele.col_min[0] + 3, ele.row_min[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(board2, (ele.center_x, ele.center_y), 10, (255, 0, 255), -1)
            # ele.draw_element(board2, put_text=text, show=False)
        # cv2.imshow('Target', board1)
        cv2.imshow('Matched Elements', board2)
        key = cv2.waitKey()
        # cv2.destroyWindow('Target')
        cv2.destroyWindow('Matched Elements')
        if match_result_save_path is not None:
            cv2.imwrite(match_result_save_path, board2)
        return key
