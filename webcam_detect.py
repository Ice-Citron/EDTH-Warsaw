from ultralytics import YOLO
import cv2

# Load your trained model (download from Colab or HuggingFace)
model = YOLO("best.pt")  # Path to your trained weights

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model(frame, conf=0.25, verbose=False)
    
    # Draw bounding boxes
    annotated_frame = results[0].plot()
    
    # Show
    cv2.imshow("Drone Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()