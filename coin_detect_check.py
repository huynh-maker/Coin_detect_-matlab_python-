import cv2
import numpy as np

cap = cv2.VideoCapture("coin_1.mp4")
ret, frame = cap.read()

# im2gray
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#adjust contrast balance
#contrast = cv2.equalizeHist(gray)
#noise reduction

bw = cv2.GaussianBlur(gray, (5,5), 0)
bw = cv2.adaptiveThreshold(bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,3, 1)




bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
#bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
# # bw = ~bw


cv2.imshow("Image processing",bw)

#BoudingBox
contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
        area = cv2.contourArea(cnt)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        #and abs(1 - area/(3.14*radius*radius)) < 0.2:
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
                cv2.putText(frame, coord_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

cv2.imshow("Image result",frame)

cv2.waitKey(0)