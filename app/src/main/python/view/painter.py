import io
import numpy as np
from PIL import Image
from PyQt5 import QtGui, QtWidgets, QtCore, uic


class Painter(QtWidgets.QWidget):
    def __init__(self, color_picker):
        super().__init__()
        self.color_picker = color_picker
        self.chosen_point = []
        self._hint = QtGui.QPixmap(500, 500)
        self._hint.fill(QtCore.Qt.white)

        dummy = np.require(
            np.zeros(shape=[500, 500, 3],
                     dtype=np.uint8), np.uint8, 'C')
        self.line_np = dummy
        dummy = QtGui.QImage(dummy, 500, 500, QtGui.QImage.Format_RGB888)
        self._line = QtGui.QPixmap(dummy)

        self.setMinimumSize(520, 520)
        self.last_x, self.last_y = None, None

        self.pen = QtGui.QPen()
        self.pen.setWidth(4)
        self.pen.setColor(QtCore.Qt.red)

    def QImageToCvMat(self, incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''

        incomingImage = incomingImage.convertToFormat(
            QtGui.QImage.Format.Format_RGBA8888)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.constBits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return arr

    def QImageToImage(self, image: QtGui.QImage):
        buf = QtCore.QBuffer()
        image.save(buf, 'png')
        return Image.open(io.BytesIO(buf.data()))

    def get_image(self):
        size = self._line.size()
        hint_map = QtGui.QPixmap(size)
        hint_map.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(hint_map)

        for pos in self.chosen_point:
            self.pen.setColor(pos['color'])
            self.pen.setWidth(pos['width'])
            painter.setPen(self.pen)
            painter.drawPoint(pos['pos'])

        painter.end()
        hint = self.QImageToImage(hint_map.toImage())
        hint = np.array(hint)
        return self.line_np, hint

    @staticmethod
    def create_pixmap(image: np.ndarray):
        image = QtGui.QImage(image, image.shape[1],
                             image.shape[0], image.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap(image)

    def set_line(self, image: Image.Image, parent) -> None:
        image = image.convert('RGB')
        w, h = image.size
        image = np.array(image)

        self.line_np = image
        self._line = self.create_pixmap(image)
        self.setFixedHeight(h)
        self.setFixedWidth(w)
        parent.resize(parent.minimumSize())

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.drawPixmap(self.rect(), self._line)
        color = self.pen.color()
        size = self.pen.width()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        for pos in self.chosen_point:
            self.pen.setColor(pos['color'])
            self.pen.setWidth(pos['width'])
            painter.setPen(self.pen)
            painter.drawPoint(pos['pos'])

        self.pen.setColor(color)
        self.pen.setWidth(size)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.LeftButton:
            data = {
                'pos': e.pos(),
                'color': self.pen.color(),
                'width': self.pen.width()
            }
            self.chosen_point.append(data)
        self.update()

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.RightButton:
            self.remove()
        self.update()

    def remove(self):
        if len(self.chosen_point) > 0:
            self.chosen_point.pop()
        self.update()
