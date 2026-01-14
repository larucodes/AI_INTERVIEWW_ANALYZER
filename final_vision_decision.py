import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    decision = "NO OBJECT"

    # ---------- RED ----------
    red1 = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255]))
    red2 = cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
    mask_red = red1 + red2

    # ---------- BLUE ----------
    mask_blue = cv2.inRange(hsv, np.array([100,150,50]), np.array([140,255,255]))

    # ---------- GREEN ----------
    mask_green = cv2.inRange(hsv, np.array([40,70,70]), np.array([80,255,255]))

    # Function to draw boxes
    def draw_box(mask, label, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                return True
        return False

    # Decision priority
    if draw_box(mask_red, "RED OBJECT", (0,0,255)):
        decision = "MOVE LEFT"
    elif draw_box(mask_blue, "BLUE OBJECT", (255,0,0)):
        decision = "MOVE RIGHT"
    elif draw_box(mask_green, "GREEN OBJECT", (0,255,0)):
        decision = "MOVE CENTER"

    # Show decision text
    cv2.putText(frame, f"Decision: {decision}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,255), 2)

    cv2.imshow("Final Vision + Decision System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
