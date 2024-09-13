import face_recognition
import argparse
import pickle
import cv2
import time  # Thêm thư viện time để tính toán thời gian

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True, help="path to the test image")
ap.add_argument("-d", "--detection_method", type=str, default="cnn", help="face detection model to use: cnn or hog")
args = vars(ap.parse_args())

# Bắt đầu tính thời gian xử lý
start_time = time.time()

# Load the known faces and encodings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# Load image và chuyển từ BGR to RGB (dlib cần để chuyển về encoding)
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Tương tự cho ảnh test: detect face, extract face ROI, chuyển về encoding
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# Khởi tạo list chứa tên các khuôn mặt phát hiện được
names = []

# Duyệt qua các encodings của faces phát hiện được trong ảnh
for encoding in encodings:
    # Khớp encoding của từng face phát hiện được với known encodings
    matches = face_recognition.compare_faces(data["encodings"], encoding, 0.4)
    name = "Unknown"

    # Kiểm tra xem từng encoding có khớp với known encodings nào không
    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        name = max(counts, key=counts.get)

    names.append(name)

# Vẽ bounding boxes và thông tin nhận diện trên ảnh
for ((top, right, bottom, left), name) in zip(boxes, names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

# Tính thời gian kết thúc
end_time = time.time()
total_time = end_time - start_time
print(f"[INFO] Thời gian xử lý ảnh: {total_time:.2f} giây")

# Hiển thị ảnh
cv2.imshow("Image", image)
cv2.waitKey(0)
