import cv2
import numpy as np
import speech_recognition as sr

# Initialize recognizer
r = sr.Recognizer()

# Camera
cap = cv2.VideoCapture(0)

system_on = False

def listen_command():
    with sr.Microphone() as source:
        print("🎙️ Listening...")
        audio = r.listen(source, phrase_time_limit=3)
        try:
            command = r.recognize_google(audio).lower()
            print("You said:", command)
            return command
        except:
            return ""

while True:

    # Listen for voice command
    command = listen_command()

    if "start" in command:
        system_on = True
        print("✅ System started")

    if "stop" in command:
        system_on = False
        print("🛑 System stopped")

    while system_on:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        decision = "NO OBJECT"

        # RED
        red1 = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255]))
        red2 = cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
        mask_red = red1 + red2

        # BLUE
        mask_blue = cv2.inRange(hsv, np.array([100,150,50]), np.array([140,255,255]))

        # GREEN
        mask_green = cv2.inRange(hsv, np.array([40,70,70]), np.array([80,255,255]))

        if cv2.countNonZero(mask_red) > 2000:
            decision = "MOVE LEFT"
        elif cv2.countNonZero(mask_blue) > 2000:
            decision = "MOVE RIGHT"
        elif cv2.countNonZero(mask_green) > 2000:
            decision = "MOVE CENTER"

        cv2.putText(frame, f"Decision: {decision}",
                    (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,255),2)

        cv2.imshow("Voice Controlled Vision System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            system_on = False
            break

cap.release()
cv2.destroyAllWindows()
