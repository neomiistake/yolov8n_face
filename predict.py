import cv2
from ultralytics import YOLO

# 載入訓練好的模型
model = YOLO(r'C:\Users\ben7u\PycharmProjects\txt_to_trainval\detect\train3\weights\best.pt')

cap = cv2.VideoCapture(0)

#迴圈讀取影像
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()#逐幀捕捉攝像頭影像。
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame,conf=0.8)
        # Visualize the results on the frame
        annotated_frame = results[0].plot() #將 YOLO 偵測到的結果直接繪製到影像上，生成帶標註的畫面。
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()