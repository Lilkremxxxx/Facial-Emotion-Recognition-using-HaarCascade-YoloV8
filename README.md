# Facial-Emotion-Recognition-using-YOLO-HaarCascade-CNN
This project builds a real-time facial emotion recognition system using CNNs. It detects faces via OpenCV and classifies seven emotions, displaying results with confidence scores. The system is lightweight, responsive, and suitable for interactive or emotion-aware applications.


Dự án này xây dựng một hệ thống nhận dạng cảm xúc khuôn mặt theo thời gian thực bằng cách sử dụng CNN. Hệ thống này phát hiện khuôn mặt thông qua OpenCV và phân loại bảy cảm xúc, hiển thị kết quả với điểm số tin cậy. Hệ thống này nhẹ, phản hồi nhanh và phù hợp với các ứng dụng tương tác hoặc nhận biết cảm xúc.

Dự án này sử dụng Dataset: FER2013 
[FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

Provided by user: Uladzislau Astrashab.

## Link đã tải sẵn Dataset: [Link](https://drive.google.com/drive/folders/1OHIsVgoL4jaUSHHpy1GgPBEK6dcDaKGr?usp=sharing)

---
# 🎭 Facial Emotion Recognition using CNN, HaarCasCade, Keras (2025)

Ứng dụng nhận diện cảm xúc khuôn mặt thời gian thực bằng CNN, hiển thị nhãn cảm xúc và độ tin cậy. Hỗ trợ ảnh tĩnh và webcam.

---

## 🧠 Mô tả / Description

- Nhận diện **7 cảm xúc cơ bản**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral  
- Xây dựng bằng **Python**, **TensorFlow/Keras**, tích hợp **OpenCV** để phát hiện khuôn mặt  
- Giao diện hiển thị bounding box màu theo cảm xúc và confidence score

---

## 🗂️ 1. Chuẩn bị môi trường / Setup

Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

---
## 📁 2. Dataset
Dữ liệu: FER2013 CSV (48×48 grayscale pixels + nhãn 7 cảm xúc)

Link tham khảo: [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

🖼️ Hình 1: Ví dụ ảnh 48×48 từ FER2013


![Screenshot 2025-06-30 135410](https://github.com/user-attachments/assets/788f1508-0dfc-4323-98f1-712534b561fd)

![Screenshot 2025-06-30 135413](https://github.com/user-attachments/assets/b770eefb-c921-4f56-b4ca-efe2554c4d33)

---

## 🧠 3. Huấn luyện mô hình / Model Training
Chuyển pixel string → mảng NumPy → reshape (48×48)

Normalize và one‑hot encode nhãn

Mạng CNN (Conv2D, MaxPool, Dropout, Dense + Softmax)

Huấn luyện ~50 epoch, lưu emotion_model.h5

---
## 🖥️ 4. Giao diện & Phân tích / GUI & Analysis
Chạy:

```bash
python face_yolo.py
```
### **Code sẽ tải những model dự đoán cảm xúc trước. Sau khi đủ model, hãy chạy lại code bắt đầu phần nhận diện.**

**Tính năng:**
Webcam: phát hiện khuôn mặt bằng YOLOv8 hoặc Haar Cascade

Tuổi & Giới tính (OpenCV DNN)

Cảm xúc (Mini‑XCEPTION)

3 cửa sổ:

 + Face Analysis: video + bounding box + info panel

 + Emotion Bars: biểu đồ thanh confidence

 + Emotion Timeline: đồ thị thời gian

🖼️ Hình 2: Ví dụ về 3 cảm xúc với 3 cửa sổ phân tích cảm xúc


![Screenshot 2025-06-30 140603](https://github.com/user-attachments/assets/e28d3df7-43c8-4000-a952-14eaf0c29b35)

![Screenshot 2025-06-30 140615](https://github.com/user-attachments/assets/38f1c120-9f7f-401c-af0e-eac51ae6c0c4)

![Screenshot 2025-06-30 140641](https://github.com/user-attachments/assets/d4325142-8439-46fb-81a6-a4765bfcb0d4)


---
## 📌 5. Ví dụ kết quả / Example Output


**Emotion 1 Neutral (0.49)**

Age: 15-20 (0.76)

Gender: Male (0.95)



**Emotion 2 Happy (0.49)**

Age: 25-32 (0.53)

Gender: Male (0.88)


--- 

## 📝 Ghi chú / Notes
Nhấn q để thoát, s để lưu ảnh, h để ẩn/hiện biểu đồ.

Có thể mở rộng thêm các model khác hoặc UI web.

---

## 👏 Cảm ơn
Đây là dự án học thuật mang tính minh họa. Dataset sử dụng từ nguồn công khai trên Kaggle: FER2013 - [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
