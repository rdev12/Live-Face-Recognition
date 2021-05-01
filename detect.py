import cv2
import face_recognition
import pickle

from numpy import average

cap = cv2.VideoCapture(0)
data = pickle.loads(open("./encodings.pickle", "rb").read())
scale = 6

while True:

    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (0, 0), fx= 1/scale, fy= 1/scale)

    boxes = face_recognition.face_locations(rgb, model="cnn")
    curr_encodings = face_recognition.face_encodings(rgb, boxes)

    if (len(boxes) == 0):
        continue
    
    distance_match = face_recognition.face_distance(data["encodings"],	curr_encodings[0])

    score = average(distance_match)

    top, right, bottom, left = boxes[0] 
    cv2.rectangle(frame, (left * scale, top * scale), (right * scale, bottom * scale), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15

    tolerance = 0.57

    if score <= tolerance:
        cv2.putText(frame, "Rishikesh", (left * scale, y * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Unknown", (left * scale, y * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()