import cv2

import numpy as np
from tensorflow.keras.models import load_model
import os
import gdown
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque
import time
from ultralytics import YOLO

class FaceAnalyzer:
    def __init__(self):
        # Đường dẫn các file mô hình
        self.age_model_path = "age_model.caffemodel"
        self.age_proto_path = "age_deploy.prototxt"
        self.gender_model_path = "gender_model.caffemodel"
        self.gender_proto_path = "deploy_gender.prototxt"
        self.emotion_model_path = "fer2013_mini_XCEPTION.110-0.65.hdf5"
        self.yolo_model_path = "yolov8n-face.pt"  # YOLOv8 nano model for face detection
        
        # Tải các mô hình nếu chưa có
        self.download_models()
        
        # Load YOLO model cho face detection
        try:
            self.face_detector = YOLO(self.yolo_model_path)
            print("YOLO face detection model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Falling back to OpenCV's Haar Cascade detector")
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.using_yolo = False
        else:
            self.using_yolo = True
        
        # Load mô hình tuổi và giới tính
        self.age_net = cv2.dnn.readNet(self.age_proto_path, self.age_model_path)
        self.gender_net = cv2.dnn.readNet(self.gender_proto_path, self.gender_model_path)
        
        # Load mô hình cảm xúc, không biên dịch lại để tránh lỗi optimizer
        self.emotion_model = load_model(self.emotion_model_path, compile=False)
        
        # Nhãn dự đoán
        self.age_labels = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
        self.gender_labels = ['Male', 'Female']
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Màu sắc cho các loại cảm xúc
        self.emotion_colors = [
            (0, 0, 255),     # Angry - Red
            (128, 0, 128),   # Disgust - Purple
            (0, 69, 255),    # Fear - Orange
            (0, 255, 0),     # Happy - Green
            (255, 0, 255),   # Sad - Magenta 
            (255, 255, 0),   # Surprise - Cyan
            (255, 255, 255)  # Neutral - White
        ]
        
        # Theo dõi cảm xúc qua thời gian
        self.emotion_history = deque(maxlen=30)  # Lưu trữ cảm xúc trong 30 khung hình
        self.fps_history = deque(maxlen=10)      # Theo dõi FPS
        
        # Thời gian bắt đầu
        self.start_time = time.time()
        self.frame_count = 0
        
        # Face tracking
        self.current_faces = {}  # dictionary to track faces
        self.next_face_id = 0    # unique ID for each new face
    
    def download_models(self):
        """Tải các mô hình nếu chưa có sẵn."""
        age_model_url = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel"
        age_proto_url = "https://raw.githubusercontent.com/steffytw/age_and_gender_prediction/master/age_deploy.prototxt"
        gender_model_url = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel"
        gender_proto_url = "https://raw.githubusercontent.com/Alialmanea/age-gender-detection-using-opencv-with-python/master/deploy_gender.prototxt"
        yolo_face_url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
        
        if not os.path.exists(self.age_model_path):
            print("Downloading age model...")
            gdown.download(age_model_url, self.age_model_path, quiet=False)
        if not os.path.exists(self.age_proto_path):
            print("Downloading age proto...")
            gdown.download(age_proto_url, self.age_proto_path, quiet=False)
        if not os.path.exists(self.gender_model_path):
            print("Downloading gender model...")
            gdown.download(gender_model_url, self.gender_model_path, quiet=False)
        if not os.path.exists(self.gender_proto_path):
            print("Downloading gender proto...")
            gdown.download(gender_proto_url, self.gender_proto_path, quiet=False)
        if not os.path.exists(self.emotion_model_path):
            print("Emotion model not found. Please place your emotion model file in the current directory.")
        if not os.path.exists(self.yolo_model_path):
            print("Downloading YOLO face detection model...")
            gdown.download(yolo_face_url, self.yolo_model_path, quiet=False)
    
    def preprocess_face(self, face, target_size=(64, 64)):
        """Tiền xử lý khuôn mặt cho mô hình cảm xúc."""
        if len(face.shape) > 2:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, target_size)
        face = face / 255.0
        face = np.expand_dims(face, axis=0)      # Kích thước: (1, 64, 64)
        face = np.expand_dims(face, axis=-1)     # Kích thước: (1, 64, 64, 1)
        return face
    
    def create_emotion_bars(self, emotion_scores):
        """Tạo biểu đồ thanh cho cảm xúc."""
        fig, ax = plt.figure(figsize=(8, 4), dpi=100), plt.gca()
        y_pos = np.arange(len(self.emotion_labels))
        
        # Chuyển đổi màu BGR sang RGB
        rgb_colors = [(r/255, g/255, b/255) for (b, g, r) in self.emotion_colors]
        
        # Vẽ biểu đồ thanh
        bars = ax.barh(y_pos, emotion_scores, color=rgb_colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.emotion_labels)
        ax.set_xlabel('Confidence')
        ax.set_title('Emotion Analysis')
        ax.set_xlim(0, 1)
        
        # Thêm giá trị trên thanh
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                    va='center')
        
        # Chuyển matplotlib figure sang mảng numpy sử dụng renderer
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        w, h = canvas.get_width_height()
        renderer = canvas.get_renderer()
        # Lấy dữ liệu ARGB và chuyển đổi sang RGB (loại bỏ kênh alpha)
        buf = np.frombuffer(renderer.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))
        rgb_image = buf[:, :, 1:4]
        plt.close(fig)
        
        return rgb_image
    
    def create_emotion_timeline(self):
        """Tạo biểu đồ đường thể hiện cảm xúc thay đổi theo thời gian."""
        if not self.emotion_history:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        fig, ax = plt.figure(figsize=(10, 5), dpi=100), plt.gca()
        data = np.array(self.emotion_history)
        x = np.arange(len(data))
        
        for i, emotion in enumerate(self.emotion_labels):
            # Chuyển đổi màu BGR sang RGB
            b, g, r = self.emotion_colors[i]
            color = (r/255, g/255, b/255)
            ax.plot(x, data[:, i], color=color, label=emotion, linewidth=2)
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Confidence')
        ax.set_title('Emotion Over Time')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right')
        ax.grid(True)
        
        # Chuyển matplotlib figure sang mảng numpy sử dụng renderer
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        w, h = canvas.get_width_height()
        renderer = canvas.get_renderer()
        buf = np.frombuffer(renderer.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))
        rgb_image = buf[:, :, 1:4]
        plt.close(fig)
        
        return rgb_image
    
    def create_info_panel(self, frame, face_count, fps):
        """Tạo bảng thông tin chung."""
        # Tạo bảng thông tin
        info_panel = np.zeros((150, frame.shape[1], 3), dtype=np.uint8)
        
        # Hiển thị thông tin
        detection_mode = "YOLO" if self.using_yolo else "Haar Cascade"
        cv2.putText(info_panel, f"Detection: {detection_mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_panel, f"Detected Faces: {face_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_panel, f"FPS: {fps:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_panel, "Press 'q': quit | 's': save | 'h': toggle views", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return info_panel
    
    def detect_faces_yolo(self, frame):
        """Phát hiện khuôn mặt sử dụng mô hình YOLO."""
        results = self.face_detector(frame, conf=0.5)  # confidence threshold of 0.5
        faces = []
        
        if results and len(results) > 0:
            for result in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, _ = result
                if conf >= 0.5:  # double check confidence
                    x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                    faces.append((x, y, w, h))
                    
        return faces
    
    def detect_faces_haar(self, frame):
        """Phát hiện khuôn mặt sử dụng Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5)
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def detect_and_analyze(self, frame):
        """Phát hiện khuôn mặt và dự đoán tuổi, giới tính và cảm xúc."""
        # Tính FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        self.fps_history.append(current_fps)
        
        # Phát hiện khuôn mặt dựa trên model đã chọn
        if self.using_yolo:
            faces = self.detect_faces_yolo(frame)
        else:
            faces = self.detect_faces_haar(frame)
        
        current_emotion_scores = np.zeros(len(self.emotion_labels))
        face_count = 0
        
        for (x, y, w, h) in faces:
            try:
                face_img = frame[y:y+h, x:x+w]
                
                # Kiểm tra kích thước khuôn mặt
                if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                    continue
                
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                            (78.4263377603, 87.7689143744, 114.895847746),
                                            swapRB=False)
                
                # Dự đoán giới tính
                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                gender = self.gender_labels[gender_preds[0].argmax()]
                gender_confidence = gender_preds[0][gender_preds[0].argmax()]
                
                # Dự đoán tuổi
                self.age_net.setInput(blob)
                age_preds = self.age_net.forward()
                age = self.age_labels[age_preds[0].argmax()]
                age_confidence = age_preds[0][age_preds[0].argmax()]
                
                # Dự đoán cảm xúc
                face_for_emotion = self.preprocess_face(face_img)
                emotion_prediction = self.emotion_model.predict(face_for_emotion, verbose=0)[0]
                emotion_label_idx = np.argmax(emotion_prediction)
                emotion = self.emotion_labels[emotion_label_idx]
                emotion_confidence = emotion_prediction[emotion_label_idx]
                
                # Cập nhật cảm xúc hiện tại (chỉ lấy từ khuôn mặt lớn nhất hoặc khuôn mặt đầu tiên)
                if face_count == 0 or w * h > face_count:
                    current_emotion_scores = emotion_prediction
                    self.emotion_history.append(emotion_prediction)
                    face_count = w * h  # Sử dụng diện tích để lưu khuôn mặt lớn nhất
                
                # Vẽ khung với màu tương ứng với cảm xúc
                emotion_color = self.emotion_colors[emotion_label_idx]
                cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_color, 2)
                
                # Hiển thị thông tin lên khuôn mặt
                info_box_width = max(w, 180)
                cv2.rectangle(frame, (x, y - 90), (x + info_box_width, y), (0, 0, 0), -1)
                
                # Thông tin tuổi, giới tính và cảm xúc
                age_text = f"Age: {age} ({age_confidence:.2f})"
                gender_text = f"Gender: {gender} ({gender_confidence:.2f})"
                emotion_text = f"Emotion: {emotion} ({emotion_confidence:.2f})"
                
                cv2.putText(frame, emotion_text, (x + 5, y - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
                cv2.putText(frame, gender_text, (x + 5, y - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, age_text, (x + 5, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Tạo biểu đồ cảm xúc nếu phát hiện khuôn mặt
        if len(faces) > 0:
            emotion_bars = self.create_emotion_bars(current_emotion_scores)
            emotion_timeline = self.create_emotion_timeline()
        else:
            # Tạo bảng trống nếu không có khuôn mặt
            emotion_bars = np.zeros((400, 800, 3), dtype=np.uint8)
            cv2.putText(emotion_bars, "No face detected", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            emotion_timeline = np.zeros((500, 1000, 3), dtype=np.uint8)
            cv2.putText(emotion_timeline, "No emotion data", (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Tạo bảng thông tin
        info_panel = self.create_info_panel(frame, len(faces), 
                                           np.mean(self.fps_history) if self.fps_history else 0)
        
        return frame, emotion_bars, emotion_timeline, info_panel

def webcam_detection():
    print("Starting professional webcam face analysis with YOLO...")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("  - Press 'h' to hide/show visualization panels")
    
    # Install required packages if not already installed
    try:
        import ultralytics
    except ImportError:
        print("Installing ultralytics package for YOLO...")
        import subprocess
        subprocess.check_call(["pip", "install", "ultralytics"])
        print("Ultralytics installed successfully.")
        import ultralytics
    
    analyzer = FaceAnalyzer()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Cài đặt kích thước cửa sổ webcam
    cv2.namedWindow('Face Analysis', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Emotion Bars', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Emotion Timeline', cv2.WINDOW_NORMAL)
    
    # Flag để hiển thị/ẩn các bảng trực quan hóa
    show_visualizations = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Phát hiện và phân tích khuôn mặt
        processed_frame, emotion_bars, emotion_timeline, info_panel = analyzer.detect_and_analyze(frame)
        
        # Ghép bảng thông tin vào khung hình chính
        complete_frame = np.vstack([processed_frame, info_panel])
        
        # Hiển thị các cửa sổ
        cv2.imshow('Face Analysis', complete_frame)
        
        if show_visualizations:
            cv2.imshow('Emotion Bars', emotion_bars)
            cv2.imshow('Emotion Timeline', emotion_timeline)
        
        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Thoát
            break
        elif key == ord('s'):  # Lưu ảnh chụp màn hình
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"face_analysis_{timestamp}.png", complete_frame)
            cv2.imwrite(f"emotion_bars_{timestamp}.png", emotion_bars)
            cv2.imwrite(f"emotion_timeline_{timestamp}.png", emotion_timeline)
            print(f"Screenshots saved with timestamp {timestamp}")
        elif key == ord('h'):  # Ẩn/hiện các bảng trực quan hóa
            show_visualizations = not show_visualizations
            if not show_visualizations:
                cv2.destroyWindow('Emotion Bars')
                cv2.destroyWindow('Emotion Timeline')
            else:
                cv2.namedWindow('Emotion Bars', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Emotion Timeline', cv2.WINDOW_NORMAL)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_detection()

import tensorflow as tf

# Giới hạn GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


