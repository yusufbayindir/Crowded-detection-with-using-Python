import cv2
import numpy as np
from ultralytics import YOLO
AREA=30 #m2
PEOPLE_THRESHOLD = 0.2  # person per m2 so in this example if there are more than 3 person in 10 m2 we give alert
CONFIDENCE_THRESHOLD = 0.1 
SLEEP_TIME = 1           
VIDEO_WIDTH = 640 
VIDEO_HEIGHT = 360

decay_rate = 0.95

max_intensity = 300

heatmap = np.zeros((300, 200), dtype=np.float32)

classNames=["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
  "teddy bear", "hair drier", "toothbrush"
  ]


model = YOLO("yolov8n.pt")



cap = cv2.VideoCapture(r"YOUR VIDEO PATH")  #If you want to use webcam feed type 0

def alert(number): 
    print("[ALERT] People count:", number)


reference_points = []
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        reference_points.append((x, y))
        cv2.circle(first_frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('First Frame - Select 4 points', first_frame)
        if len(reference_points) == 4:
            cv2.destroyAllWindows()


ret, first_frame = cap.read()
if not ret:
    print("Failed to grab the first video frame.")
    cap.release()
    exit()


cv2.imshow('First Frame - Select 4 points', first_frame)
cv2.setMouseCallback('First Frame - Select 4 points', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()


if len(reference_points) != 4:
    print("You must select exactly 4 points.")
    cap.release()
    exit()


image_points = np.array(reference_points, dtype='float32')

#real world coordinates
real_world_points = np.array([[0, 0], [0, 300], [200, 300], [200, 0]], dtype='float32')
#homography matrix
H, status = cv2.findHomography(image_points, real_world_points)

if H is None or not status.all():
    print("Homography calculation was not successful.")
    cap.release()
    exit()

# Function to process detections, draw bounding boxes and labels, and return coordinates
def processResults(results, frame):
    people_coordinates = []  
    for r in results:
        boxes = r.boxes
        peopleCount = 0
        for box in boxes:
            if box.conf < CONFIDENCE_THRESHOLD:
                continue
            cls_id = int(box.cls[0])
            if classNames[cls_id] == "person":
                peopleCount += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x, center_y = (x1 + x2) // 2, y2
                people_coordinates.append((center_x, center_y))  # Store bottom center coordinates
                # Draw bounding box around detected person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put the class name on top of the bounding box
                text = f"{classNames[cls_id]}: {box.conf[0]:.2f}"
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
        if peopleCount >= PEOPLE_THRESHOLD*AREA: 
            alert(peopleCount)
    return people_coordinates




def generate_heatmap(canvas, intensity=10):
    normalized_canvas = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX)
    incremented_canvas = np.clip(normalized_canvas * intensity, 0, 255)
    heatmap = cv2.applyColorMap(np.uint8(incremented_canvas), cv2.COLORMAP_JET)
    return heatmap

def update_heatmap(bird_eye_points):
    global heatmap
    heatmap *= decay_rate
    for x, y in bird_eye_points:
        heatmap[int(y), int(x)] += 1  # Increment the count at the location

    #apply Gaussian blur to the heatmap 
    heatmap_blurred = cv2.GaussianBlur(heatmap, (15, 15), 0)

    minVal, maxVal, _, _ = cv2.minMaxLoc(heatmap_blurred)
    if maxVal != 0:
        heatmap_blurred = heatmap_blurred / maxVal
    return heatmap_blurred

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 
    
    #object detection
    results = model(frame)
    

    people_coordinates = processResults(results , frame)
    person_points = np.array(people_coordinates, dtype='float32').reshape(-1, 1, 2)

    #homography transformation
    if status.all():
        person_points_bird_view = cv2.perspectiveTransform(person_points, H)

        #update heatmap with bird's eye view points
        for point in person_points_bird_view:
            x, y = point.ravel()
            if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
                cv2.circle(heatmap, (int(x), int(y)), 3, max_intensity, -1)  

        heatmap = np.clip(heatmap * decay_rate, 0, 300) 

        # Rotate and flip the heatmap to reach camera angle after the perpective tranformation 
        heatmap_rotated = np.rot90(heatmap)
        heatmap_mirrored = np.flipud(heatmap_rotated)

        heatmap_blurred = cv2.GaussianBlur(heatmap_mirrored, (11, 11), 0)

        heatmap_normalized = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX)

        colored_heatmap = cv2.applyColorMap(np.uint8(heatmap_normalized), cv2.COLORMAP_JET)

        cv2.imshow('Heatmap', colored_heatmap)

    #display the video
    cv2.imshow('Frame', frame)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
