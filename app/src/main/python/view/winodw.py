from PIL import Image
from PyQt5 import QtGui, QtWidgets, QtCore, uic
from inference import predict

from . import Painter, ErrLogDialog

import platform

op_sys = platform.system()
err_log_dialog = None


def popup_err_dialog(title: str = "Err!", contents: str = ""):
    global err_log_dialog
    if err_log_dialog is None:
        err_log_dialog = ErrLogDialog()

    err_log_dialog.exec_(title, contents)


class Window(QtWidgets.QMainWindow):
    def show(self) -> None:
        super(Window, self).show()

    def __init__(self, cfx):
        super(Window, self).__init__()
        self.ctx = cfx

        uic.loadUi(self.ctx.main_ui, self)
        self.eBtn.setIcon(QtGui.QIcon(self.ctx.eraser_icon))

        self.color_picker = QtWidgets.QColorDialog()
        self.painter = Painter(self.color_picker)
        self.verticalLayout_3.addWidget(self.painter)

        self.resize(self.minimumSize())
        screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor().pos())
        self.move(screen.size().width() // 3, screen.size().height() // 4)
        self.setWindowTitle("")
        self.setFixedSize(self.size())
        self.__change_color(None)
        self.setAcceptDrops(True)

        self.__event_init()

        self.update()

    def __change_color(self, color):
        if color is None:
            color = self.color_picker.currentColor()

        rgb = list()
        rgba = color.getRgb()

        for index in range(3):
            rgb.append(rgba[index])
        color = QtGui.QColor(rgb[0], rgb[1], rgb[2])
        self.colorBtn.setStyleSheet("color:black;"
                                    "border-style: outset;"
                                    "border-width: 1px;"
                                    "border-radius: 5px;"
                                    "border-color: black;"
                                    "background-color: %s;" % color.name())
        self.painter.pen.setColor(color)
        self.update()

    def __event_init(self):
        self.__liner_flag = False
        self.colorBtn.clicked.connect(self.__color_btn_clicked)
        self.eBtn.clicked.connect(self.painter.remove)
        self.runBtn.clicked.connect(self.__run_btn_clicked)
        self.fileOpen.triggered.connect(self.__file_open)
        self.fileSave.triggered.connect(self.__file_save)
        self.penSizeSlider.valueChanged.connect(
            lambda size: self.__set_pan_size(size))

        self.status.setText("HI")

    def __color_btn_clicked(self):
        color = self.color_picker.getColor()
        self.__change_color(color)

    def __set_pan_size(self, size=2):
        self.penSizeLabel.setText(str(size))
        self.painter.pen.setWidth(size)

    def __pen_btn_clicked(self):
        self.penSizeLabel.setText(str(2))
        self.painter.setpen(pen_size=2, color=QtCore.Qt.black)

    def __get_pred_image(self):
        _, hint = self.painter.get_image()
        self.__status_update(1)
        line = self.origin_line
        hint = Image.fromarray(hint)
        img = predict(line, hint, None)

        if not isinstance(img, Image.Image):
            popup_err_dialog('Inference Err', img)

        return img

    def __run_btn_clicked(self):
        img = self.__get_pred_image()
        self.__status_update("Done")
        img.show()

    def __liner_btn_clicked(self):
        pass

    def __imread(self, path):
        img = Image.open(path)
        return img.convert('RGB')

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def wheelEvent(self, e: QtGui.QWheelEvent):
        modifiers = QtGui.QGuiApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            pen_size = self.penSizeSlider.value()
            delta = e.angleDelta().y()
            if delta > 0:
                pen_size += 1
            if delta < 0:
                pen_size -= 1
            self.penSizeSlider.setValue(pen_size)

    def dropEvent(self, e):
        if e.mimeData().hasUrls:
            e.setDropAction(QtCore.Qt.CopyAction)
            e.accept()
            for url in e.mimeData().urls():
                file_name = str(url.toLocalFile())
            self.__file_open(file_name)
        else:
            e.ignore()

    def __file_open(self, file_path=None):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileters = self.tr(
            "Image Files (*.png *.jpg *.bmp *.jpeg *.JPG *.PNG *.JPEG)")
        if file_path is None:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Open File', None, fileters, options=options)
            if file_path == "":
                return -1

        try:
            img = self.__imread(file_path)
        except Exception:
            self.__file_open()
            return -1

        if 'png' in file_path.lower():
            img = img.convert('LA')

        self.origin_line = img.copy()

        width = float(img.size[0])
        height = float(img.size[1])

        if width > height:
            rate = width / height
            new_height = 512
            new_width = int(512 * rate)
        else:
            rate = height / width
            new_width = 512
            new_height = int(rate * 512)

        img = img.resize((new_width, new_height), Image.BICUBIC)
        self.painter.chosen_point.clear()
        self.painter.set_line(img, self)
        self.setFixedSize(new_width + 14, new_height + 170)
        self.update()
        self.__liner_flag = False

    def _imwrite(self, image: Image):
        try:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                'Save File',
                None,
                "Image Files (*.png *.jpg *.bmp *.jpeg *.JPG *.PNG *.JPEG)",
                options=options)

            if file_path == "":
                return -1
            image.save(file_path)
        except Exception:
            self._imwrite(image)

    def __file_save(self):
        image = self.__get_pred_image()
        if image == -1:
            return -1
        self.__status_update(8)
        self.__status_update(10)
        self._imwrite(image)
        self.__status_update("Done")

    def __status_update(self, message):
        self.status.setText(str(message))
        QtGui.QGuiApplication.processEvents()
        self.update()
