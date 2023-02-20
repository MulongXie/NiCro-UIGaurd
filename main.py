from utils.ModelLoader import modelLoader
from NiCro import NiCro

ml= modelLoader()
ml.load_models()

nicro = NiCro(ocr_opt='google', dp_model_loader=ml)

nicro.load_devices()
nicro.get_devices_info()

nicro.load_robot()

nicro.select_source_device(1)

nicro.record_and_replay_actions(output_root='/home/ml/Data/visual testing/testcase',
                                app_name='Desktop',
                                testcase_id='1',
                                is_record=False,
                                is_replay=True)
