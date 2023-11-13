import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

# hàm load_labels dùng để trả về danh sách các nhãn của lablesPath (cụ thể là file yolov3.txt)


def load_labels(labelsPath):
    return open(labelsPath).read().strip().split("\n")

# Hàm này nhận đầu vào là số lượng lớp đối tượng num_classes và trả về một mảng 2D chứa các giá trị màu sắc ngẫu nhiên cho mỗi lớp.


def load_colors(num_classes):
    return np.random.randint(0, 255, size=(num_classes, 3), dtype="uint8")

# để tải mô hình từ dạng Darknet nhận đầu vào là đường dẫn configPath(yolov3.cfg) và weightsPath(yolov3.weights)
# trả về là một đối tượng mạng neural đã được tải.


def load_model(configPath, weightsPath):
    return cv2.dnn.readNetFromDarknet(configPath, weightsPath)


'''hàm tiền xử lí hình ảnh, nhận vào hình ảnh và kích thước và trả về một đối tượng 
    blob(ảnh đã đc xử lí) và kích thước gốc của ảnh '''


def preprocess_image(image, target_size):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, target_size, swapRB=True, crop=False)
    return blob, (H, W)


''' hàm phát hiện đối tượng trả về một tuple gồm các danh sách boxes, confidences, classIDs 
    chứa thông tin về các đối tượng được phát hiện '''


def detect_objects(net, blob, output_layers, confidence_threshold, W, H):
    net.setInput(blob)
    layerOutputs = net.forward(output_layers)
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    return boxes, confidences, classIDs

# áp dụng thuật toán NMS để lọc dữ liệu trùng và cho ra dữ liệu chính xác hơn


def apply_nms(boxes, confidences, confidence_threshold, nms_threshold):
    idxs = cv2.dnn.NMSBoxes(
        boxes, confidences, confidence_threshold, nms_threshold)
    return idxs

# vẽ các dự đoán lên ảnh gốc dựa trên thông tin về các dữ liệu đã lọc, độ tin cậy,nhãn, màu sắc...


def draw_predictions(image, boxes, confidences, classIDs, labels, colors):
    if len(boxes) > 0:
        for i in range(len(boxes)):
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# hàm xử lí hình ảnh và show ra hình ảnh đã xử lí


def process_image(image_path):
    labelsPath = 'yolov3.txt'
    LABELS = load_labels(labelsPath)
    COLORS = load_colors(len(LABELS))
    weightsPath = 'yolov3.weights'
    configPath = 'yolov3.cfg'
    net = load_model(configPath, weightsPath)
    image = cv2.imread(image_path)
    blob, (H, W) = preprocess_image(image, (416, 416))
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    boxes, confidences, classIDs = detect_objects(
        net, blob, ln, 0.5, W, H)
    idxs = apply_nms(boxes, confidences, 0.5, 0.3)
    draw_predictions(image, np.array(boxes)[idxs], np.array(confidences)[
        idxs], np.array(classIDs)[idxs], LABELS, COLORS)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.imwrite("temp/object-detection.jpg", image)
    cv2.destroyAllWindows()

# hàm nhận biết đối tượng thông qua webcam


def process_webcam():
    labelsPath = 'yolov3.txt'
    LABELS = load_labels(labelsPath)
    COLORS = load_colors(len(LABELS))
    weightsPath = 'yolov3.weights'
    configPath = 'yolov3.cfg'
    net = load_model(configPath, weightsPath)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        blob, (H, W) = preprocess_image(frame, (416, 416))
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        boxes, confidences, classIDs = detect_objects(
            net, blob, ln, 0.5, W, H)
        idxs = apply_nms(boxes, confidences, 0.5, 0.3)
        draw_predictions(frame, np.array(boxes)[idxs], np.array(confidences)[
            idxs], np.array(classIDs)[idxs], LABELS, COLORS)
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# giao diện


def choose_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)


def choose_webcam():
    process_webcam()


root = tk.Tk()
root.geometry("300x200")
root.resizable(False, False)
frame_label = tk.Frame(root)
frame_label.pack(pady=30)
label_text = "Object Detection"
label_font = ("Arial", 16)
label_color = "blue"
label = tk.Label(frame_label, text=label_text,
                 font=label_font, foreground=label_color)
label.pack()
button_frame = tk.Frame(root)
button_frame.pack()
button1 = tk.Button(button_frame, text="Using Image",
                    width=10, height=2, command=choose_image)
button1.pack(side="left", padx=10)
button2 = tk.Button(button_frame, text="Using Camera",
                    width=10, height=2, command=choose_webcam)
button2.pack(side="left", padx=10)
root.mainloop()
