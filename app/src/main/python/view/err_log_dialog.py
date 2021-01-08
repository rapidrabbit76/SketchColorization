from PyQt5 import QtGui, QtWidgets, QtCore, uic


class ErrLogDialog(QtWidgets.QDialog):
    def __init__(self):
        super(ErrLogDialog, self).__init__()
        self.__status_label = QtWidgets.QTextEdit()
        self.__status_label.setStyleSheet(self.styleSheet())
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.__status_label)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)

    def exec_(self, title: str = "Err!", contents: str = "") -> int:
        self.setWindowTitle(title)
        self.__status_label.setText(contents)
        super(ErrLogDialog, self).exec_()
        return -1

    def hideEvent(self, a0: QtGui.QHideEvent) -> None:
        self.__status_label.clear()
