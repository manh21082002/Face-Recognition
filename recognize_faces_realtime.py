import face_recognition
import argparse
import pickle
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encodings")
ap.add_argument("-d", "--detection_method", type=str, default="hog", help="face detection model to use: cnn or hog")  # Sử dụng 'hog' cho thời gian thực nhanh hơn
args = vars(ap.parse_args())

# Load the known faces and encodings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# Mở webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Bắt đầu tính thời gian xử lý khung hình
    start_time = time.time()

    # Đọc khung hình từ webcam
    ret, frame = video_capture.read()

    # Chuyển đổi khung hình từ BGR sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện các khuôn mặt và tính toán encoding của các khuôn mặt trong khung hình
    boxes = face_recognition.face_locations(rgb_frame, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb_frame, boxes)

    # Khởi tạo danh sách các tên khuôn mặt được phát hiện
    names = []

    # Duyệt qua các encoding của khuôn mặt trong khung hình
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding, 0.4)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    # Vẽ bounding boxes và tên của khuôn mặt trên khung hình
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Tính thời gian xử lý và hiển thị tốc độ xử lý (FPS)
    end_time = time.time()
    total_time = end_time - start_time
    fps = 1 / total_time
    print(f"[INFO] FPS: {fps:.2f}")

    # Hiển thị khung hình
    cv2.imshow("Video", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng cửa sổ
video_capture.release()
cv2.destroyAllWindows()
