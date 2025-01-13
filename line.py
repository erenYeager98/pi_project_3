import sys
import time
import numpy as np
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import io
from threading import Condition
import RPi.GPIO as GPIO
import subprocess

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(9, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(13, GPIO.OUT)
pwm = GPIO.PWM(13, 10000)
pwm.start(0)

# Streaming Output Class
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

# Displacement Calculation Thread
class DisplacementThread(QThread):
    displacement_signal = pyqtSignal(float, float)

    def __init__(self, camera_app):
        super().__init__()
        self.camera_app = camera_app
        self.running = True

    def run(self):
        while self.running:
            time.sleep(2)  # Run every 2 seconds
            if self.camera_app.background_image is not None:
                start_time = time.time()
                dx, dy = self.camera_app.calculate_displacement(
                    self.camera_app.background_image, self.camera_app.current_image
                )
                calculation_time = (time.time() - start_time) * 1000
                print(f"Calculation Time: {calculation_time:.2f} ms")
                self.displacement_signal.emit(dx, dy)

    def stop(self):
        self.running = False

# Main Application Class
class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration(main={"size": (320, 240)}))
        self.output = StreamingOutput()
        self.picam2.start_recording(JpegEncoder(), FileOutput(self.output))

        self.setWindowTitle("Camera with GPIO Controls")
        self.setGeometry(100, 100, 640, 480)

        # UI Elements
        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)
        self.displacement_label = QLabel("Displacement: ΔX = 0 cm, ΔY = 0 cm", self)
        self.displacement_label.setAlignment(Qt.AlignCenter)
        # self.capture_button = QPushButton("Capture Image", self)
        # self.capture_button.clicked.connect(self.capture_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.displacement_label)
        # layout.addWidget(self.capture_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.background_image = None
        self.overlay_image = None
        self.current_image = np.zeros((240, 320, 3), dtype=np.uint8)

        # Start displacement calculation thread
        self.displacement_thread = DisplacementThread(self)
        self.displacement_thread.displacement_signal.connect(self.update_displacement_ui)
        self.displacement_thread.start()

        # ORB Configuration
        self.orb = cv2.ORB_create(nfeatures=300)

        # Monitor GPIO in another timer
        self.gpio_timer = QTimer()
        self.gpio_timer.timeout.connect(self.check_gpio)
        self.gpio_timer.start(100)

    def update_frame(self):
        with self.output.condition:
            self.output.condition.wait()
            frame_data = np.frombuffer(self.output.frame, dtype=np.uint8)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

        if self.overlay_image is not None:
            overlay_resized = cv2.resize(self.overlay_image, (frame.shape[1], frame.shape[0]))
            frame = cv2.addWeighted(overlay_resized, 0.6, frame, 0.4, 0)

        height, width, channel = frame.shape
        qimage = QImage(frame.data, width, height, 3 * width, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage))
        self.current_image = frame

    def check_gpio(self):
        if GPIO.input(9) == GPIO.HIGH:
            self.reset_overlay()
            self.capture_image()
            
    def capture_image(self):
        self.background_image = self.current_image.copy()
        self.overlay_image = self.background_image.copy()
        print("Image Captured and Overlay Applied")

    def reset_overlay(self):
        self.overlay_image = None
        self.background_image = None
        print("Overlay and Effects Reset")
        self.displacement_label.setText("Displacement: ΔX = 0 cm, ΔY = 0 cm")

    def update_displacement_ui(self, dx, dy):
        self.displacement_label.setText(f"Displacement: ΔX = {dx:.2f} cm, ΔY = {dy:.2f} cm")

    def calculate_displacement(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return 0, 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if matches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
            displacement = np.mean(dst_pts - src_pts, axis=0)
            pixel_to_cm = 0.05
            return displacement[0] * pixel_to_cm, displacement[1] * pixel_to_cm
        else:
            return 0, 0

    def closeEvent(self, event):
        pwm.stop()
        GPIO.cleanup()
        self.picam2.stop_recording()
        self.displacement_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
