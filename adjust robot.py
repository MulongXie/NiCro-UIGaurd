from utils.ModelLoader import modelLoader

ml= modelLoader()
ml.load_models()

from NiCro import NiCro

nicro = NiCro(ocr_opt='google', dp_model_loader=ml)
nicro.load_robot()
nicro.robot.control_robot_by_clicking_on_cam_video()   # test the robot system