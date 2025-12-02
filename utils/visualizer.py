# utils/visualizer.py
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QPainter, QColor
import numpy as np
class Visualizer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = np.zeros(128)
        self.setFixedHeight(120)
        self.setStyleSheet("background-color:#000;")
    def update_data(self, v):
        self.data = np.roll(self.data, -1)
        self.data[-1] = v
        self.repaint()
    def paintEvent(self, event):
        pix = QPixmap(self.width(), self.height())
        pix.fill(QColor("#000"))
        p = QPainter(pix)
        w = pix.width()/len(self.data)
        for i,h in enumerate(self.data):
            color = QColor.fromHsv((i*3)%360,255,255)
            p.fillRect(int(i*w), pix.height()-int(h), int(w-1), int(h), color)
        p.end()
        self.setPixmap(pix)