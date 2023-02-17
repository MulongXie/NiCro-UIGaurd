import os, json, cv2
from glob import glob
# from nms import nms_for_results_bbox
import torch
import torch.nn as nn
from torchvision import transforms
import time
time.clock = time.time

import sys
sys.path.insert(0, r"C:\Mulong\Code\Demo\finalCode-UIED\GUI-Semantics-main")

from darkpattern.gather_basic_info import get_color_status_icon
from darkpattern.template_matching.template_matching import get_ad_icons, TemplateMatching
from darkpattern.merge_tm_checkgroup import merge_tm_results_checkgroup
from darkpattern.rule_check import predict_type

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 31695120968


class UIGuard:
	def __init__(self, model_loader):
		self.dpCode2dpRealName = {"II-AM-G-SMALL": "Interface Inference",
								"FA-G-WATCHAD": "Forced Action",
								"SN-FC": "Forced Continuity",
								"II-AM-TWE": "Aesthetic Manipulation",
								"II-AM-DA": "Disguised Ad",
								"FA-SOCIALPYRAMID": "Social Pyramid",
								"FA-GAMIFICATION": "Forced Action",
								"NG-RATE": "Nag to rate",
								"NG-UPGRADE": "Nag to upgrade",
								"FA-G-COUNTDOWNAD": "Forced Action",
								"II-AM-FH": "False Hierarchy",
								"FA-G-PRO": "Forced Action",
								"II-PRE-FOLLOW": "Preselection",
								"II-PRE-NOTIFICATION": "Preselection",
								"II-PRE-PRIVACY": "Preselection",
								"II-PRE-USAGE-DATA": "Preselection",
								"II-PRE": "Preselection",
								"FA-Privacy": "Forced Action",
								}

		self.model_loader = model_loader
		self.template_matcher = model_loader.template_matcher

	def resize_bbox(self, dets, img_w, img_h):
		for det in dets:
			curr_bbox = det["bbox"]
			scale = img_h/800
			det["bbox"] = [int(co*scale) for co in curr_bbox]
		return dets

	def extract_property(self, image_path, img_cv, elements_info, vis=False):
		merged_dets_nms = elements_info

		h,w,_ = img_cv.shape
		merged_dets_nms = self.resize_bbox(merged_dets_nms, w, h)

		# bbox is 800 h
		# ------
		# get colors, checkbox, icon semantic results
		model_loader = self.model_loader
		gather_info = get_color_status_icon(merged_dets_nms, image_path, model_loader.transform_test, model_loader.device,
											model_loader.model_icon, model_loader.class_name_icon, model_loader.model_status, model_loader.class_name_status)
		# get ad icons
		ad_icons_close, ad_icons_info = get_ad_icons(img_cv, self.template_matcher, vis=vis)
		# add ad icons to gather info
		# merge checkbox with its text and make them a check_group
		all_properties = merge_tm_results_checkgroup(gather_info, ad_icons_close, ad_icons_info)
		return all_properties

	def darkpatternChecker(self, all_properties, img_cv):
		'''
		:return output: {dp_type: [{"label": dp_type, "bbox":[x1,y1,x2,y2], "subType": desc,
						"flag": str, "text": str, "fg_lum": float, "bg_lum":float,
						"con":float, "bg_color", [r,g,b], "fg_color":[r,g,b],
						"children": [{"category":Button, "bbox":[x1,y1,x2,y2], "score": float, "matched": bool, "text_items": [...]}]}]}
		'''
		img_h, img_w, _ = img_cv.shape
		final_results = predict_type(all_properties, img_h, img_w)
		return final_results

	def detect_dark_pattern(self, image_path, elements_info, vis=False):
		start_time = time.time()
		img_cv = cv2.imread(image_path)
		all_properties = self.extract_property(image_path, img_cv, elements_info, vis)
		final_results = self.darkpatternChecker(all_properties, img_cv)
		android_output = self.organise_output_for_android(final_results)
		print("Reorganise results Using {:.02f}s".format(time.time() - start_time))
		print("++ android_output", android_output)
		return android_output

	def organise_output_for_android(self, output):
		android_output = {"results":[]}
		for each_dp_type, item_list in output.items():
			for item in item_list:
				typ_ = self.dpCode2dpRealName.get(each_dp_type, each_dp_type)
				tmp_object = {"type": typ_, "desc": item["subType"].split("]")[1], "text_content": item.get("text", ""), "container_bbox": item["bbox"], "children": []}
				for child in item.get("children", []):
					tmp_object["children"].append({"bbox": child["bbox"]})
				android_output["results"].append(tmp_object)
		return android_output


# if __name__ == '__main__':
# 	# main()
# 	uiguard = UIGuard()
#
#
# 	test_data_root = "Twitter"
# 	all_test_data = glob(test_data_root + "/noti/**.jpg")
# 	# test_data_root = "/Users/che444/Desktop/DPCODE-CLEAN/finalCode/data/annotations/Rico_testset/testset"
# 	# all_test_data = glob(test_data_root + "/**.jpg")
#
# 	output_root = os.path.join(test_data_root, "noti/detection")
# 	if not os.path.exists(output_root):
# 		os.makedirs(output_root)
#
# 	start_time = time.time()
# 	for image_path in tqdm(all_test_data):
# 		# if "20177" not in image_path:
# 		# 	continue
# 		uiguard.UIGuard(image_path, output_root, vis=False)
# 	end_time = time.time()
# 	print(f"Using {(end_time-start_time)/len(all_test_data)}/img")









