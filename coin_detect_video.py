import cv2
import numpy as np

# Input & output video
video_path = "coin_2.mp4"
output_path = "coin_detected_2.mp4"

# Đọc video đầu vào
cap = cv2.VideoCapture(video_path)

# Lấy thông tin video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Khởi tạo ghi video đầu ra
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #noise reduction
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 1)

    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    #bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
    
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if area >1000 and abs(1 - area/(3.14*radius*radius)) < 0.2:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            
            # Tính centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                coord_text = f"({cx}, {cy})"
                
                # Hiển thị tọa độ
                cv2.putText(frame, coord_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
    
    # Ghi frame ra file
    out.write(frame)
    
    # Hiển thị real-time (bấm q để thoát)
    cv2.imshow("Coin Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Coin detection completed. Output saved.")
