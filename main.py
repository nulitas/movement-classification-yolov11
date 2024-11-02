import cv2
from ultralytics import YOLO

model = YOLO("best.pt")


cap = cv2.VideoCapture("classroom.mp4")


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    results = model(frame, conf=0.6, classes=[0, 8])

    for result in results:
        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy.squeeze())
            conf = box.conf[0]
            label = box.cls[0]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            

            cv2.putText(
                frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
            )


    cv2.imshow("YOLO Predictions", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
