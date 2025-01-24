import sys
import time
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import io
from threading import Condition
import RPi.GPIO as GPIO
import subprocess
from collections import deque

GPIO.setmode(GPIO.BCM)
GPIO.setup(9, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(13, GPIO.OUT)
pwm = GPIO.PWM(13, 10000)
pwm.start(0)

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class DisplacementThread(QThread):
    displacement_signal = pyqtSignal(float, float, float) 

    def __init__(self, camera_app):
        super().__init__()
        self.camera_app = camera_app
        self.running = True

    def run(self):
        while self.running:
            if self.camera_app.background_image is not None:
                start_time = time.time()
                dx, dy = self.camera_app.calculate_displacement(
                    self.camera_app.background_image, self.camera_app.current_image
                )
                calculation_time = (time.time() - start_time) * 1000
                if dx is not None and dy is not None:
                    self.displacement_signal.emit(dx, dy, calculation_time)

    def stop(self):
        self.running = False

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration(main={"size": (320, 240)}))
        self.output = StreamingOutput()
        self.picam2.start_recording(JpegEncoder(), FileOutput(self.output))
        self.shift_threshold_cm = 1.0  
        self.previous_dx_mm = 0.0
        self.previous_dy_mm = 0.0
        self.setWindowTitle("App")
        self.setGeometry(100, 100, 640, 480)

        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)
        self.displacement_label = QLabel("Displacement: ΔX = 0 mm, ΔY = 0 mm", self)
        self.displacement_label.setAlignment(Qt.AlignCenter)
        self.calculation_time_label = QLabel("Calculation Time: 0 ms", self)
        self.calculation_time_label.setAlignment(Qt.AlignCenter)
        self.pwm_duty_cycle_label = QLabel("PWM Duty Cycle: 0%", self)
        self.pwm_duty_cycle_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.displacement_label)
        layout.addWidget(self.calculation_time_label)
        layout.addWidget(self.pwm_duty_cycle_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.background_image = None
        self.overlay_image = None
        self.current_image = np.zeros((240, 320, 3), dtype=np.uint8)

        self.displacement_thread = DisplacementThread(self)
        self.displacement_thread.displacement_signal.connect(self.update_displacement_ui)
        self.displacement_thread.start()

        self.orb = cv2.ORB_create(nfeatures=1000)

        self.displacement_history = deque(maxlen=5)

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
            self.capture_image()
        elif GPIO.input(17) == GPIO.HIGH:
            self.shutdown_pi()

    def shutdown_pi(self):
        subprocess.run(['sudo', 'shutdown', 'now'])

    def capture_image(self):
        self.reset_overlay()
        self.update_frame()
        self.background_image = self.current_image.copy()
        self.overlay_image = self.background_image.copy()
        print("Image Captured and Overlay Applied")

    def reset_overlay(self):
        self.overlay_image = None
        self.background_image = None
        print("Overlay and Effects Reset")
        self.displacement_label.setText("Displacement: ΔX = 0 mm, ΔY = 0 mm")

    def update_displacement_ui(self, dx, dy, calculation_time):
        self.displacement_history.append((dx, dy))
        avg_dx = np.mean([d[0] for d in self.displacement_history])
        avg_dy = np.mean([d[1] for d in self.displacement_history])
        self.displacement_label.setText(f"Displacement: ΔX = {avg_dx:.2f} mm, ΔY = {avg_dy:.2f} mm")
        self.calculation_time_label.setText(f"Calculation Time: {calculation_time:.2f} ms")
        self.update_pwm(avg_dx, avg_dy)

    def update_pwm(self, dx, dy):
        if dy <= -10:
            duty_cycle = 0
        elif dy >= 10:
            duty_cycle = 100
        else:
            duty_cycle = (dy + 10) * 5 

        pwm.ChangeDutyCycle(duty_cycle)
        self.pwm_duty_cycle_label.setText(f"PWM Duty Cycle: {duty_cycle:.2f}%")


    def calculate_displacement(self, img1, img2):
            if img1 is not None and img2 is not None:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                orb = cv2.ORB_create()
                kp1, des1 = orb.detectAndCompute(gray1, None)
                kp2, des2 = orb.detectAndCompute(gray2, None)

                if des1 is None or des2 is None:
                    return None, None

                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)

                if len(matches) > 0:
                    matches = sorted(matches, key=lambda x: x.distance)
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

                    displacement = np.mean(dst_pts - src_pts, axis=0)
                    dx, dy = displacement

                    pixel_to_mm_factor = 0.5
                    dx_mm = dx * pixel_to_mm_factor
                    dy_mm = dy * pixel_to_mm_factor

                    if abs(dx_mm - self.previous_dx_mm) >= self.shift_threshold_cm:
                        self.previous_dx_mm = dx_mm
                    else:
                        dx_mm = self.previous_dx_mm

                    if abs(dy_mm - self.previous_dy_mm) >= self.shift_threshold_cm:
                        self.previous_dy_mm = dy_mm
                    else:
                        dy_mm = self.previous_dy_mm

                    return dx_mm, dy_mm
                else:
                    return None, None
            else:
                return None, None
                
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
