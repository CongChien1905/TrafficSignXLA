import cv2
import numpy as np
from keras.models import load_model

# Load model đã huấn luyện
model = load_model('my_model.h5')

# Dictionary ánh xạ nhãn lớp
classes = {
    1: 'Speed limit (20km/h)',
    2: 'Speed limit (30km/h)',
    3: 'Speed limit (50km/h)',
    4: 'Speed limit (60km/h)',
    5: 'Speed limit (70km/h)',
    6: 'Speed limit (80km/h)',
    7: 'End of speed limit (80km/h)',
    8: 'Speed limit (100km/h)',
    9: 'Speed limit (120km/h)',
    10: 'No passing',
    11: 'No passing veh over 3.5 tons',
    12: 'Right-of-way at intersection',
    13: 'Priority road',
    14: 'Yield',
    15: 'Stop',
    16: 'No vehicles',
    17: 'Veh > 3.5 tons prohibited',
    18: 'No entry',
    19: 'General caution',
    20: 'Dangerous curve left',
    21: 'Dangerous curve right',
    22: 'Double curve',
    23: 'Bumpy road',
    24: 'Slippery road',
    25: 'Road narrows on the right',
    26: 'Road work',
    27: 'Traffic signals',
    28: 'Pedestrians',
    29: 'Children crossing',
    30: 'Bicycles crossing',
    31: 'Beware of ice/snow',
    32: 'Wild animals crossing',
    33: 'End speed + passing limits',
    34: 'Turn right ahead',
    35: 'Turn left ahead',
    36: 'Ahead only',
    37: 'Go straight or right',
    38: 'Go straight or left',
    39: 'Keep right',
    40: 'Keep left',
    41: 'Roundabout mandatory',
    42: 'End of no passing',
    43: 'End no passing veh > 3.5 tons'
}

def preprocess_image(image):
    """
    Xử lý ảnh để phù hợp với mô hình:
    - Resize về 30x30.
    - Chuẩn hóa giá trị pixel về [0, 1].
    """
    image = cv2.resize(image, (30, 30))
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch
    image = image / 255.0  # Chuẩn hóa
    return image

def predict_traffic_sign(image):
    """
    Dự đoán loại biển báo giao thông.
    """
    processed_image = preprocess_image(image)
    pred_probabilities = model.predict(processed_image)[0]
    pred_class = np.argmax(pred_probabilities) + 1  # Lớp bắt đầu từ 1
    return classes.get(pred_class, "Unknown sign"), pred_probabilities[pred_class - 1]

# Mở webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ webcam.")
        break

    # Lấy ROI (vùng quan tâm) từ khung hình
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    offset = 100  # Kích thước vùng nhận diện
    roi = frame[center_y - offset:center_y + offset, center_x - offset:center_x + offset]

    try:
        # Dự đoán biển báo từ ROI
        label, confidence = predict_traffic_sign(roi)
        confidence_text = f"{confidence * 100:.2f}%"  # Độ tự tin
    except Exception as e:
        label = "Error"
        confidence_text = ""

    # Hiển thị nhãn và khung ROI trên khung hình
    cv2.rectangle(frame, (center_x - offset, center_y - offset), (center_x + offset, center_y + offset), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} ({confidence_text})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Hiển thị khung hình
    cv2.imshow("Traffic Sign Recognition", frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
