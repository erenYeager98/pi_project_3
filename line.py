import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from picamera2 import Picamera2

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        self.camera = Picamera2()
        self.camera.start()

        self.setWindowTitle("Camera Preview")
        self.setGeometry(100, 100, 640, 480)
        self.layout = QVBoxLayout()

        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        self.capture_button = QPushButton("Capture", self)
        self.capture_button.clicked.connect(self.capture_image)
        self.layout.addWidget(self.capture_button)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) 

        self.setLayout(self.layout)

    def update_frame(self):
        frame = self.camera.capture_array()
        
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)

    def capture_image(self):
        filename = "captured_image.jpg"
        self.camera.capture_file(filename)
        print(f"Image saved as {filename}")

    def closeEvent(self, event):
        self.camera.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
