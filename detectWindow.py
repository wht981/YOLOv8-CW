import sys
import cv2
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, 
                             QWidget, QHBoxLayout, QFileDialog, QTextEdit, QGroupBox, QGridLayout)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO

class YoloAppWhite(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Settings ---
        self.model_path = 'yolov8n.pt'
        self.model = None
        # ----------------

        self.setWindowTitle("Detection System")
        self.setGeometry(100, 100, 1300, 800)
        self.setStyleSheet("background-color: #f5f5f5;") # Set main window background to light gray

        # Variables
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_running = False
        self.current_source = None

        # UI Setup
        self.init_ui()
        self.load_model(self.model_path)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20) # Add margins around the edges
        main_layout.setSpacing(20)

        # --- Top: Display Area ---
        display_layout = QHBoxLayout()
        display_layout.setSpacing(20)
        
        # Style for the display labels (White background, gray border, rounded corners)
        display_style = """
            QLabel {
                border: 2px solid #cccccc; 
                border-radius: 10px; 
                background-color: white; 
                color: #888888; 
                font-size: 20px;
            }
        """

        # 1. Left: Input Source
        source_layout = QVBoxLayout()
        
        self.label_source = QLabel("Waiting for Input...")
        self.label_source.setAlignment(Qt.AlignCenter)
        self.label_source.setStyleSheet(display_style)
        self.label_source.setMinimumSize(600, 450)
        self.label_source.setScaledContents(True)
        
        source_layout.addWidget(self.label_source)

        # 2. Right: Detection Result
        result_layout = QVBoxLayout()

        self.label_result = QLabel("Waiting for Result...")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setStyleSheet(display_style)
        self.label_result.setMinimumSize(600, 450)
        self.label_result.setScaledContents(True)

        result_layout.addWidget(self.label_result)

        display_layout.addLayout(source_layout)
        display_layout.addLayout(result_layout)
        
        # --- Bottom: Controls ---
        bottom_layout = QHBoxLayout()
        
        # Control Panel Group
        control_group = QGroupBox("Control Panel")
        control_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 8px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        control_grid = QGridLayout()
        control_grid.setVerticalSpacing(10)
        control_grid.setHorizontalSpacing(10)

        # Button Style
        btn_style = """
            QPushButton {
                background-color: white;
                border: 1px solid #bbbbbb;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """
        
        self.btn_load = QPushButton("Load Model")
        self.btn_cam = QPushButton("Camera")
        self.btn_img = QPushButton("Image")
        self.btn_vid = QPushButton("Video")
        self.btn_stop = QPushButton("Stop / Reset")
        
        # Apply styles
        for btn in [self.btn_load, self.btn_cam, self.btn_img, self.btn_vid, self.btn_stop]:
            btn.setStyleSheet(btn_style)
        
        # Specific style for Stop button (Light Red)
        self.btn_stop.setStyleSheet(btn_style.replace("background-color: white;", "background-color: #ffebee; color: #c62828; border: 1px solid #ef9a9a;"))

        # Connect signals
        self.btn_load.clicked.connect(self.select_model)
        self.btn_cam.clicked.connect(self.start_camera)
        self.btn_img.clicked.connect(self.select_image)
        self.btn_vid.clicked.connect(self.select_video)
        self.btn_stop.clicked.connect(self.stop_detection)

        # Layout buttons
        control_grid.addWidget(self.btn_load, 0, 0)
        control_grid.addWidget(self.btn_cam, 0, 1)
        control_grid.addWidget(self.btn_img, 1, 0)
        control_grid.addWidget(self.btn_vid, 1, 1)
        control_grid.addWidget(self.btn_stop, 2, 0, 1, 2)
        
        control_group.setLayout(control_grid)

        # Console
        console_group = QGroupBox("System Log")
        console_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 8px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        console_layout = QVBoxLayout()
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("border: none; background-color: transparent; font-family: Consolas; font-size: 12px;")
        console_layout.addWidget(self.console)
        console_group.setLayout(console_layout)

        bottom_layout.addWidget(control_group, stretch=1)
        bottom_layout.addWidget(console_group, stretch=2)

        main_layout.addLayout(display_layout, stretch=4)
        main_layout.addLayout(bottom_layout, stretch=1)

        central_widget.setLayout(main_layout)

    # --- Logic ---
    def log(self, msg):
        t = time.strftime("%H:%M:%S")
        self.console.append(f"[{t}] {msg}")
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())
        print(f"[{t}] {msg}")

    def load_model(self, path):
        self.log(f"Loading model: {path}...")
        try:
            self.model = YOLO(path)
            self.model_path = path
            self.log("Model loaded.")
        except Exception as e:
            self.log(f"Error: {e}")

    def select_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Pt Files (*.pt)")
        if f: self.load_model(f)

    def stop_detection(self):
        self.timer.stop()
        if self.cap: self.cap.release()
        self.cap = None
        self.is_running = False
        
        # Reset Displays to Text
        self.label_source.clear()
        self.label_source.setText("Waiting for Input...")
        self.label_result.clear()
        self.label_result.setText("Waiting for Result...")
        self.log("Stopped.")

    def start_camera(self):
        self.stop_detection()
        self.current_source = 'cam'
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.timer.start(30)
            self.is_running = True
            self.log("Camera started.")
        else:
            self.log("Camera error.")

    def select_video(self):
        self.stop_detection()
        f, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video (*.mp4 *.avi *.mkv)")
        if f:
            self.current_source = 'vid'
            self.cap = cv2.VideoCapture(f)
            self.timer.start(30)
            self.is_running = True
            self.log(f"Video: {f}")

    def select_image(self):
        self.stop_detection()
        f, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image (*.jpg *.png)")
        if f:
            self.log(f"Image: {f}")
            img = cv2.imread(f)
            if img is not None:
                self.display(img, self.label_source)
                res = self.model(img)[0]
                self.display(res.plot(), self.label_result)
                self.log(f"Objects: {len(res.boxes)}")

    def update_frame(self):
        if not self.cap or not self.is_running: return
        ret, frame = self.cap.read()
        if not ret:
            if self.current_source == 'vid': self.stop_detection()
            return
        
        self.display(frame, self.label_source)
        res = self.model(frame, verbose=False)[0]
        self.display(res.plot(), self.label_result)

    def display(self, img, label):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c*w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YoloAppWhite()
    window.show()
    sys.exit(app.exec_())