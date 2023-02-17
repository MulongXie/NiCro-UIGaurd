from utils.GUI import GUI
from utils.UIGuard import UIGuard
from utils.ModelLoader import modelLoader

ml= modelLoader()
ml.load_models()

gui = GUI('data/input/0.jpg', model_loader=ml)
gui.detect_element(True, True, True, ocr_opt='google')
gui.classify_compos()

dp = UIGuard(model_loader=ml)
dp.detect_dark_pattern(image_path=gui.img_path, elements_info=gui.get_elements_info_ui_guard(), vis=False)
