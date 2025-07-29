from deepface import DeepFace
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = frame[y:y+h, x:x+w]


        result = DeepFace.analyze(img_path=roi_gray, actions=['emotion'],enforce_detection=False)


        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


        cv2.putText(frame, f"{result[0]['dominant_emotion']}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Live Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





