from darkpattern.UIGuard import UIGuard


class DarkPattern(UIGuard):
    def __init__(self, gui):
        super().__init__()
        self.gui = gui
        self.report = []

    def detect_dark_pattern(self):
        '''
        :param gui: util.GUI object, containing the element detection and compo classification info
        :return boolean
        '''
        elements_info = self.gui.get_elements_info_ui_guard()
        self.report = self.UIGuard(image_path=self.gui.img_path,  elements_info=elements_info, vis=False)

