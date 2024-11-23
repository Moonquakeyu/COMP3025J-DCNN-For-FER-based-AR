import cv2  
import numpy as np  
from keras.models import load_model  

# 加载训练好的模型  
model = load_model('path_to_your_model.h5')  

# 初始化摄像头  
cap = cv2.VideoCapture(0)  

# 加载Haar Cascade人脸检测模型  
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  

# 开始视频流  
while True:  
    ret, frame = cap.read()  
    if not ret:  
        continue  

    # 将捕获的帧转换为灰度图  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

    # 检测人脸  
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  

    for (x, y, w, h) in faces:  
        # 在检测到的人脸周围画一个矩形  
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  

        # 提取脸部区域并调整大小  
        face = gray[y:y+h, x:x+w]  
        face = cv2.resize(face, (48, 48))  

        # 预处理图像（归一化）  
        face = face / 255.0  
        face = np.expand_dims(face, axis=0)  
        face = np.expand_dims(face, axis=-1)  # 将形状调整为 (1, 48, 48, 1)  

        # 预测情感  
        emotion_prediction = model.predict(face)  
        emotion_label = np.argmax(emotion_prediction)  # 得到预测的情感标签  

        # 在图像上显示预测的情感  
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)  

    # 显示结果帧  
    cv2.imshow('Emotion Recognition', frame)  

    # 按 'q' 键退出  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  

# 释放摄像头并关闭所有窗口  
cap.release()  
cv2.destroyAllWindows()