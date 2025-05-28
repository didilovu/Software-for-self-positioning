# main.py - основной файл приложения
import sys
import cv2
import re
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
from ui_design import Ui_MainWindow
from modul import point_proc, wheremypoint, far, camera_position, path, map_to_mat

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.capture = None
        self.timer = QTimer()
        self.model = YOLO('best.pt')
        self.arr_point = point_proc('points.txt')
        self.current_cam_point = None
        
        self.ui.mode_combo.currentIndexChanged.connect(self.on_combobox_changed)
        self.ui.upload_btn.clicked.connect(self.load_image)
        self.timer.timeout.connect(self.process_video_frame)
        self.ui.coord_input.returnPressed.connect(self.calculate_path)
        self.on_combobox_changed()

    def on_combobox_changed(self):
        selected_text = self.ui.mode_combo.currentText()
        self.update_content(selected_text)
        
    def update_content(self, selected_option):
        if selected_option == "Изображение":
            self.ui.upload_btn.setVisible(True)
            self.ui.image_label.setVisible(True)
            self.stop_webcam()
            self.ui.image_label.clear()
            
        elif selected_option == "Видео":
            self.ui.upload_btn.setVisible(False)
            self.ui.image_label.setVisible(True)
            self.start_webcam()
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_path:
            frame = cv2.imread(file_path)
            if frame is not None:
                processed_frame = self.process_frame(frame)
                self.display_image(processed_frame)
            else:
                print("Не удалось загрузить изображение")
    
    def start_webcam(self):       
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Не удалось открыть камеру")
            return
        self.timer.start(30)
    
    def stop_webcam(self):
        if self.capture and self.capture.isOpened():
            self.capture.release()
        self.timer.stop()
        self.capture = None
    
    def display_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        self.ui.image_label.setPixmap(pixmap.scaled(
            self.ui.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
    
    def closeEvent(self, event):
        self.stop_webcam()
        event.accept()
    
    def process_video_frame(self):
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                processed_frame = self.process_frame(frame)
                self.display_image(processed_frame)
            
    def process_frame(self, frame):
        results = self.model(frame)
        distances = {}
        
        if results:
            boxes = results[0].boxes
            for box in boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                if conf > 0.6:
                    X, Y = wheremypoint(x1, y1, x2, y2)
                    cls = int(cls)
                    
                    color = (0, 0, 255) if cls == 0 else (255, 0, 0)
                    cv2.circle(frame, (int(X), int(Y)), 15, color, -1)
                    cv2.putText(frame, f"Class {cls}", (int(x1), int(y1)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    z = far(y2 - y1) * 100
                    distances[cls] = z
            
            if 0 in distances and 1 in distances:
                cam_point = camera_position(
                    self.arr_point[0], 
                    self.arr_point[1], 
                    distances[0], 
                    distances[1]
                )
                self.current_cam_point = cam_point  # Сохраняем текущую позицию камеры
                cv2.putText(frame, f"Camera: {cam_point}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Обновляем label_3 с координатами камеры
                self.update_camera_position_label()
        return frame
    
    def update_camera_position_label(self):
        """Обновляет label_3 с текущей позицией камеры"""
        if self.current_cam_point:
            self.ui.label_3.setText(
                f"Текущая позиция камеры:\n"
                f"X: {self.current_cam_point[0]:.2f}\n"
                f"Y: {self.current_cam_point[1]:.2f}"
            )
    
    def calculate_path(self):
        """Вычисляет маршрут между текущей позицией камеры и введенными координатами"""
        if not self.current_cam_point:
            self.ui.label_3.setText("Сначала определите позицию камеры!")
            return
            
        input_text = self.ui.coord_input.text()

        coords = re.findall(r'[-+]?\d*\.\d+|\d+', input_text)
        if len(coords) == 2:
            end_point = (float(coords[0]), float(coords[1]))
                
            df = map_to_mat(self.arr_point, self.current_cam_point, 400)
            route_points = path(df,self.current_cam_point, end_point)
            route_text = "Маршрут:\n"
            for i, point in enumerate(route_points, 1):
                route_text += f"{i}. X: {point[0]:.2f}, Y: {point[1]:.2f}\n"
                    
            self.ui.label_3.setText(
                f"Начальная точка:\n"
                f"X: {self.current_cam_point[0]:.2f}\n"
                f"Y: {self.current_cam_point[1]:.2f}\n\n"
                f"Конечная точка:\n"
                f"X: {end_point[0]:.2f}\n"
                f"Y: {end_point[1]:.2f}\n\n"
                f"{route_text}"
                )  
        else:
            self.ui.label_3.setText("Неверный формат координат!\nИспользуйте: X,Y")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())