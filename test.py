from utils.GUI import GUI
from element_detection.classify_compo.CNN import CNN
from utils.DarkPattern import DarkPattern

gui = GUI('data/input/0.jpg')
gui.detect_element(True, True, True, ocr_opt='google')
gui.load_model()
gui.classify_compos()

dp = DarkPattern(gui)
dp.load_models()
dp.detect_dark_pattern()
