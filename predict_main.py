from PyQt4 import QtGui, Qt, QtCore
import sys
import multiprocessing
from py_paddle import swig_paddle as api
from py_paddle import DataProviderConverter
from paddle.trainer.PyDataProvider2 import dense_vector
import numpy as np

PREDICT_PASS = 3


def paddle_predict_main(q, result_q):
    api.initPaddle("--use_gpu=false")
    gm = api.GradientMachine.loadFromConfigFile("./output/model/pass-00000/trainer_config.py")
    assert isinstance(gm, api.GradientMachine)
    converter = DataProviderConverter(input_types=[dense_vector(28 * 28)])
    while True:
        features = q.get()
        val = gm.forwardTest(converter([[features]]))[0]['value'][0]
        result_q.put(val)


class PaintWidget(QtGui.QWidget):
    def __init__(self, zoom_size=16, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.setSizePolicy(Qt.QSizePolicy.Expanding,
                           Qt.QSizePolicy.Expanding)
        self.setMinimumSize(zoom_size * 28, zoom_size * 28)
        self.setMaximumSize(zoom_size * 28, zoom_size * 28)
        self.is_painting = False
        self.lines = []
        self.last_pos = None
        self.zoom_size = zoom_size
        pen = QtGui.QPen(Qt.QColor.fromRgb(0))
        pen.setWidth(self.zoom_size * 2)
        self.pen = pen
        self.setStyleSheet("background-color: white;")

    def mousePressEvent(self, ev):
        assert isinstance(ev, QtGui.QMouseEvent)
        self.is_painting = True
        self.last_pos = ev.pos()
        ev.accept()

    def mouseMoveEvent(self, ev):
        assert isinstance(ev, QtGui.QMouseEvent)
        if self.is_painting:
            self.lines.append(
                (self.last_pos, ev.pos()))
            self.last_pos = ev.pos()
            self.repaint()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        self.is_painting = False
        self.lines.append((self.last_pos, ev.pos()))
        self.repaint()
        ev.accept()

    def paintEvent(self, ev):
        assert isinstance(ev, QtGui.QPaintEvent)
        painter = QtGui.QPainter(self)
        painter.setPen(self.pen)
        for pos in self.lines:
            painter.drawLine(*pos)

    def get_image_feature(self):
        img = QtGui.QPixmap(self.zoom_size * 28, self.zoom_size * 28)
        self.render(img)
        img = img.toImage()
        assert isinstance(img, QtGui.QImage)
        for y in xrange(28):
            for x in xrange(28):
                cnt = 0
                for offset_y in xrange(self.zoom_size):
                    for offset_x in xrange(self.zoom_size):
                        pixel = img.pixel(x * self.zoom_size + offset_x,
                                          y * self.zoom_size + offset_y)
                        if pixel == 0xff000000:
                            cnt += 1
                yield float(cnt) / float(self.zoom_size ** 2)

    @QtCore.pyqtSlot()
    def clear(self):
        self.lines = []
        self.repaint()


class MainWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.main_layout = QtGui.QVBoxLayout()
        self.paint_widget = PaintWidget()
        self.main_layout.addWidget(self.paint_widget)
        self.btn_layout = QtGui.QHBoxLayout()
        self.clear_btn = QtGui.QPushButton(u'Clear')
        self.btn_layout.addWidget(self.clear_btn)
        self.predict_btn = QtGui.QPushButton(u'Predict')
        self.btn_layout.addWidget(self.predict_btn)
        self.main_layout.addLayout(self.btn_layout)
        self.setLayout(self.main_layout)
        self.clear_btn.clicked.connect(self.paint_widget.clear)
        self.predict_btn.clicked.connect(self.predict)

        self.queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.paddle_process = multiprocessing.Process(
            target=paddle_predict_main, args=(self.queue, self.result_queue)
        )
        self.paddle_process.start()

    @QtCore.pyqtSlot()
    def predict(self):
        fea = list(self.paint_widget.get_image_feature())

        for row in xrange(28):
            def map_fea_to_mono(x):
                return 1 if x != 0.0 else 0

            print map(map_fea_to_mono, fea[row * 28: (row + 1) * 28])

        self.queue.put(fea)
        val = self.result_queue.get()
        QtGui.QMessageBox.warning(self, 'Predict Result', u'Predict Result is %d with prob %f%%' % (
            np.argmax(val), max(val)
        ))

    def closeEvent(self, ev):
        self.paddle_process.terminate()
        ev.accept()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    widget = MainWidget()
    widget.show()
    app.exec_()
