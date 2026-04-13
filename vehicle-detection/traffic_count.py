import cv2
import numpy as np

cap = cv2.VideoCapture('data/videos/cars.mp4')

# 取得影片寬高, 縮放比例
cap_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  

scale = 0.5
new_w  = int(cap_w * scale)
new_h = int(cap_h * scale)

# 建立背景分離器 (MOG2)
fgbg = cv2.createBackgroundSubtractorMOG2()

# 形態學運算用 kernel
kernel = np.ones((5, 5), np.uint8)

# 計數線位置
count = 0
line_y = new_h // 2

while True:
    ret, frame = cap.read()

    if not ret:
        print("讀取影片失敗")
        break
    if frame is None:
        break
    
    # 縮小影像
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 前景偵測
    fgmask = fgbg.apply(frame)

    # 去雜訊：開運算 + 膨脹
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)
    
    # 二值化：刪陰影
    _, fgmask = cv2.threshold(fgmask , 200, 255, cv2.THRESH_BINARY)

    # 找輪廓
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 畫計數線
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 過濾太小的雜訊
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)

            # 計算車輛中心點
            cx = x + w // 2
            cy = y + h // 2

            # 畫矩形框 + 中心點
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # 判斷是否穿過計數線
            if line_y - 5 < cy < line_y + 5:
                count += 1

    # 顯示計數結果
    cv2.putText(frame, f'Count: {count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 220), 2)

    # 顯示畫面與前景 mask
    cv2.imshow('frame', frame)
    cv2.imshow('mask', fgmask)

    # 按 ESC 離開
    if cv2.waitKey(50) == 27:
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
