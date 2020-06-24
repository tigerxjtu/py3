from PyQt5.QtWidgets import QMainWindow, QFrame, QDesktopWidget, QApplication, QWidget, QMenu, QAction, QFileDialog,QScrollArea
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal, QRect
from PyQt5.QtGui import QPainter, QColor, QPixmap, QImage, QCursor
import sys

class ImageView(QWidget):
    zoomRequest = pyqtSignal(int)
    scrollRequest = pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        super(ImageView,self).__init__(*args, **kwargs)
        self.pixmap = QPixmap()
        self.zoom_value = 1.0
        self.pt_interval_x = 0
        self.pt_interval_y = 0
        self.pressed = False
        self.prev_pos = None
        self.prev_cur = None

    def load_image(self):
        try:
            img_path,_ = QFileDialog.getOpenFileName(self,"Open Image","./","Images(*.png *.jpg *.gif)")
            print(img_path)
            if img_path:
                image = QImage(img_path)
                self.pixmap = QPixmap.fromImage(image)
                self.rest()
                self.update()
        except Exception as e:
            print(e)

    def paintEvent(self, event):
        try:
            if not self.pixmap or self.pixmap.isNull():
                return super(ImageView, self).paintEvent(event)
            painter = QPainter(self)
            width = min(self.pixmap.width(), self.width())
            height = width*1.0/(self.pixmap.width()*1.0/self.pixmap.height())
            height = min(height, self.height())
            width = height*(self.pixmap.width()/self.pixmap.height())

            painter.translate(self.width()//2 + self.pt_interval_x, self.height()//2+self.pt_interval_y)
            painter.scale(self.zoom_value,self.zoom_value)

            rect = QRect(-width//2, -height//2, width, height)
            painter.drawPixmap(rect, self.pixmap)
        except Exception as e:
            print(e)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.prev_pos = event.pos()#point.x(),point.y()
            self.pressed = True
            self.prev_cur = self.cursor()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:  # 这里只能用buttons(), 因为button()在mouseMoveEvent()中无论
            if not self.pressed:
               return super(ImageView,self).mouseMoveEvent(event)
            self.setCursor(Qt.SizeAllCursor)
            pos = event.pos()
            dx = pos.x() - self.prev_pos.x()
            dy = pos.y() - self.prev_pos.y()
            self.pt_interval_x += dx
            self.pt_interval_y += dy
            self.prev_pos = pos
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.pressed:
                self.pressed = False
                self.prev_pos = None
                self.setCursor(self.prev_cur)
                self.prev_cur = None
                self.pt_interval_x = 0
                self.pt_interval_y = 0

    def wheelEvent(self, event):
        delta = event.angleDelta()
        value = delta.y()
        if value>0:
            self.zoom_in()
        elif value<0:
            self.zoom_out()

        # mods = event.modifiers()
        # if Qt.ControlModifier == int(mods) and value:
        #     self.zoomRequest.emit(value)
        # else:
        #     value and self.scrollRequest.emit(value, Qt.Vertical)
        #     delta.x() and self.scrollRequest.emit(delta.x(), Qt.Horizontal)
        if value:
            self.zoomRequest.emit(self.zoom_value)

        event.accept()

    def contextMenuEvent(self, event):
        # pos = event.pos();
        # pos = self.mapFromGlobal(pos)
        menu = QMenu(self)

        load_img = QAction("Load Image")
        menu.addAction(load_img)
        load_img.triggered.connect(self.load_image)
        zoom_in = QAction("Zoom In")
        menu.addAction(zoom_in)
        zoom_in.triggered.connect(self.zoom_in)
        zoom_out = QAction("Zoom Out")
        menu.addAction(zoom_out)
        zoom_out.triggered.connect(self.zoom_out)
        zoom_reset = QAction("Zoom Reset")
        menu.addAction(zoom_reset)
        zoom_reset.triggered.connect(self.rest)

        menu.exec(self.mapToGlobal(event.pos()))

    def rest(self):
        self.zoom_value=1.0
        self.pt_interval_x = 0
        self.pt_interval_y = 0
        self.update()

    def zoom_in(self):
        self.zoom_value += 0.2
        self.adjustSize()
        self.update()

    def zoom_out(self):
        self.zoom_value -= 0.2
        self.adjustSize()
        self.update()

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.zoom_value * self.pixmap.size()
        return super(ImageView, self).minimumSizeHint()

class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        '''initiates application UI'''

        # w = QWidget()
        # self.setCentralWidget(w)
        self.tboard = ImageView(self)
        # self.tboard.setMinimumSize(540, 720)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.tboard)
        self.scroll.setWidgetResizable(True)

        self.scrollBars = {
            Qt.Vertical: self.scroll.verticalScrollBar(),
            Qt.Horizontal: self.scroll.horizontalScrollBar()
        }
        self.tboard.zoomRequest.connect(self.zoom_request)

        self.setCentralWidget(self.scroll)

        self.statusbar = self.statusBar()
        # self.tboard.msg2Statusbar[str].connect(self.statusbar.showMessage)

        self.resize(540,720)
        self.center()
        self.setWindowTitle('Image Viewer')
        self.show()

    def center(self):
        '''centers the window on the screen'''

        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)


    def zoom_request(self,delta):
        # self.tboard.adjustSize()
        return
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)
        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)



if __name__ == '__main__':
    app = QApplication([])
    image_viewer = MainWin()
    sys.exit(app.exec_())