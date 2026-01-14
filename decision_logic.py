import cv2
import numpy as np

cap = cv2.VideoCapture(0)

decision = "NO OBJECT"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # RED
    red1 = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255]))
    red2 = cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
    mask_red = red1 + red2

    # BLUE
    mask_blue = cv2.inRange(hsv, np.array([100,150,50]), np.array([140,255,255]))

    # GREEN
    mask_green = cv2.inRange(hsv, np.array([40,70,70]), np.array([80,255,255]))

    # Priority-based decision
    if cv2.countNonZero(mask_red) > 2000:
        decision = "MOVE LEFT (RED)"
    elif cv2.countNonZero(mask_blue) > 2000:
        decision = "MOVE RIGHT (BLUE)"
    elif cv2.countNonZero(mask_green) > 2000:
        decision = "MOVE CENTER (GREEN)"
    else:
        decision = "NO OBJECT"

    # Show decision
    cv2.putText(frame, f"Decision: {decision}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    cv2.imshow("Decision Logic", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
