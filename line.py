import sys
import time
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import io
from threading import Condition, Thread
import RPi.GPIO as GPIO
import subprocess

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(9, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class GPIOShutdownThread(Thread):
    def run(self):
        while True:
            if GPIO.input(17) == GPIO.HIGH:
                print("Shutdown button pressed. Shutting down...")
                subprocess.run(["sudo", "shutdown", "now"])

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Configure camera for lower resolution (320x240) to speed up processing
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration(main={"size": (320, 240)}))
        self.output = StreamingOutput()
        self.picam2.start_recording(JpegEncoder(), FileOutput(self.output))

        self.setWindowTitle("Camera Preview with GPIO Controls")
        self.setGeometry(100, 100, 640, 480)

        # Camera display area
        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)

        # Displacement calculation time display
        self.displacement_label = QLabel("Displacement calculation time: 0 ms", self)

        # Capture button
        self.capture_button = QPushButton("Capture Image", self)
        self.capture_button.clicked.connect(self.capture_image)

        # Layouts
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.displacement_label)  # Add the displacement time label
        layout.addWidget(self.capture_button)  # Add the capture button

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timers for updating the image and checking GPIO
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.shutdown_thread = GPIOShutdownThread()
        self.shutdown_thread.start()

        self.background_image = None
        self.overlay_image = None
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.check_gpio_9)
        self.capture_timer.start(100)

        # Limit the number of features in ORB for faster computation
        self.orb = cv2.ORB_create(nfeatures=300)  # Limiting features

    def update_frame(self):
        with self.output.condition:
            self.output.condition.wait()
            frame_data = np.frombuffer(self.output.frame, dtype=np.uint8)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

        if frame is not None:
            if self.overlay_image is not None:
                overlay_resized = cv2.resize(self.overlay_image, (frame.shape[1], frame.shape[0]))
                frame = cv2.addWeighted(overlay_resized, 0.6, frame, 0.4, 0)

            height, width, channel = frame.shape
            qimage = QImage(frame.data, width, height, 3 * width, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qimage))
            self.current_image = frame

    def check_gpio_9(self):
        if GPIO.input(9) == GPIO.HIGH:
            start_time = time.time()
            while GPIO.input(9) == GPIO.HIGH:
                if (time.time() - start_time) > 1.0:
                    return

            self.capture_image()

    def capture_image(self):
        current_image = self.current_image.copy()
        if self.background_image is not None:
            start_time = time.time()
            dx, dy = self.calculate_displacement(self.background_image, current_image)
            displacement_time = (time.time() - start_time) * 1000  # Convert to ms
            self.displacement_label.setText(f"Displacement calculation time: {displacement_time:.2f} ms")
            print(f"Displacement: ΔX = {dx:.2f} cm, ΔY = {dy:.2f} cm")

        self.background_image = current_image

    def calculate_displacement(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Feature detection using ORB
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return 0, 0

        # Using FLANN for faster matching
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        bf = cv2.FlannBasedMatcher(index_params, search_params)
        matches = bf.match(des1, des2)

        if matches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
            displacement = np.mean(dst_pts - src_pts, axis=0)

            dx, dy = displacement
            pixel_to_cm = 0.05
            return dx * pixel_to_cm, dy * pixel_to_cm
        else:
            return 0, 0

    def is_frame_significantly_different(self, img1, img2, threshold=50):
        diff = cv2.absdiff(img1, img2)
        non_zero_count = np.count_nonzero(diff)
        return non_zero_count > threshold

    def closeEvent(self, event):
        self.picam2.stop_recording()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
