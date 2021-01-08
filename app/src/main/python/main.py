import sys

import multiprocessing
import os
import qdarkstyle
from fbs_runtime.application_context.PyQt5 import ApplicationContext, cached_property

from view import Window

import inference


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

multiprocessing.set_start_method('forkserver', force=True)
multiprocessing.freeze_support()


class AppContext(ApplicationContext):
    __MAIN_UI__ = 'MAINUI.ui'
    __SETTINGDIALOG_UI__ = 'dialog.ui'
    __ERASER_ICON__ = 'eraser.png'
    __MODEL_PATH__ = 'SketchColorizationModel.onnx'

    def __init__(self):
        super(AppContext, self).__init__()

    @cached_property
    def main_ui(self):
        return self.get_resource(self.__MAIN_UI__)

    @cached_property
    def eraser_icon(self):
        return self.get_resource(self.__ERASER_ICON__)

    @cached_property
    def model_path(self):
        return self.get_resource(self.__MODEL_PATH__)


if __name__ == '__main__':
    appctxt = AppContext()
    window = Window(appctxt)
    window.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.show()
    inference.__HANDLER__ = inference.InferenceHandler(appctxt)
    exit_code = appctxt.app.exec_()
    sys.exit(exit_code)
