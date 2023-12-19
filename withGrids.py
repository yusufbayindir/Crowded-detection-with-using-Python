import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
AREA=30 #m2
PEOPLE_THRESHOLD = 0.3  # person per m2
CONFIDENCE_THRESHOLD = 0.1  # only show if equal or greater than this
SLEEP_TIME = 1            # ms
VIDEO_WIDTH = 640 # TODO : auto detect
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

# Initialize video capture
cap = cv2.VideoCapture(r"C:\Users\Joseph\Desktop\aaaaas.mp4")  # Replace with your video path

# Function to handle mouse clicks and collect points
reference_points = []
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        reference_points.append((x, y))
        cv2.circle(first_frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('First Frame - Select 4 points', first_frame)
        if len(reference_points) == 4:
            cv2.destroyAllWindows()

# Grab the first frame of the video to select points
ret, first_frame = cap.read()
if not ret:
    print("Failed to grab the first video frame.")
    cap.release()
    exit()

# Display the first frame and set the mouse callback function to capture points
cv2.imshow('First Frame - Select 4 points', first_frame)
cv2.setMouseCallback('First Frame - Select 4 points', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Make sure that four points have been selected
if len(reference_points) != 4:
    print("You must select exactly 4 points.")
    cap.release()
    exit()

# Convert the list of tuples to a numpy array
image_points = np.array(reference_points, dtype='float32')

# Define real-world coordinates for the selected points here
real_world_points = np.array([[0, 0], [0, 300], [200, 300], [200, 0]], dtype='float32')

# Calculate the Homography matrix
H, status = cv2.findHomography(image_points, real_world_points)

# Check if the homography matrix is valid
if H is None or not status.all():
    print("Homography calculation was not successful.")
    cap.release()
    exit()

# Function to process detections and return coordinates
# Function to process detections, draw bounding boxes and labels, and return coordinates
def alert(number): 
    print("[ALERT] People count:", number)
def processResults(results, frame):
    people_coordinates = []  # Initialize an empty list to store coordinates
    for r in results:
        peopleCount = 0
        boxes = r.boxes
        for box in boxes:
            if box.conf < CONFIDENCE_THRESHOLD:
                continue
            cls_id = int(box.cls[0])
            if classNames[cls_id] == "person":
                peopleCount += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                people_coordinates.append((center_x, y2))  # Store bottom center coordinates
                # Draw bounding box around detected person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put the class name on top of the bounding box
                text = f"{classNames[cls_id]}: {box.conf[0]:.2f}"
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if peopleCount >= PEOPLE_THRESHOLD*AREA: 
            alert(peopleCount)
    return people_coordinates

def generate_heatmap(canvas, intensity=10):
    # Scale the canvas to the range 0-255
    normalized_canvas = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX)
    # Increment the point's intensity to make it visible on the heatmap
    incremented_canvas = np.clip(normalized_canvas * intensity, 0, 255)
    # Apply the color map to the incremented canvas
    heatmap = cv2.applyColorMap(np.uint8(incremented_canvas), cv2.COLORMAP_JET)
    return heatmap

def update_heatmap(bird_eye_points):
    global heatmap
    # Decay the heatmap to fade out old points
    heatmap *= decay_rate
    
    # Update the heatmap with new points
    for x, y in bird_eye_points:
        heatmap[int(y), int(x)] += 1  # Increment the count at the location

    # Apply Gaussian blur to the heatmap for smoother visualization
    heatmap_blurred = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # Normalize the heatmap
    minVal, maxVal, _, _ = cv2.minMaxLoc(heatmap_blurred)
    if maxVal != 0:
        heatmap_blurred = heatmap_blurred / maxVal
    return heatmap_blurred

# In order to generate grids in a frame 
def generateGrids(image, image_to_show, pointsArray):
    height, width, _ = image.shape                                          #Shape of exact frame
    height_image_to_show, width_image_to_show, _ = image_to_show.shape      #Shape of heatmap

    rows = 3
    cols = 3        # rows * cols = grid amount 

    grid_color = (0, 255, 0)  # Green
    grid_thickness = 2

    row_spacing = height // rows                    
    row_spacing_image_to_show=height_image_to_show//rows            # In order to obtain equal grid rows
    col_spacing = width // cols         
    col_spacing_image_to_show=width_image_to_show//cols             # In order to obtain equal grid cols

    # Draws horizontal lines for grid 
    for i in range(1, rows):
        y = i * row_spacing_image_to_show
        cv2.line(image_to_show, (0, y), (width_image_to_show, y), grid_color, grid_thickness)

    # Draws vertical lines for grid 
    for j in range(1, cols):
        x = j * col_spacing_image_to_show
        cv2.line(image_to_show, (x, 0), (x, height_image_to_show), grid_color, grid_thickness)
    
    grid_counts = countPeopleInGrids(pointsArray, rows, cols, row_spacing, col_spacing)         
    printGridCountsWithText(grid_counts, row_spacing_image_to_show, col_spacing_image_to_show, image_to_show)
    cv2.imshow('Frame', image_to_show)

def countPeopleInGrids(people_coordinates, rows, cols, row_spacing, col_spacing):
    # Create dictionary and assign 0 to all values of grid keys
    grid_counts = {f"{i+1}": 0 for i in range(rows * cols)}

    # Increase each determined people center into the relevant grid.
    for person in people_coordinates:
        x, y = person
        grid_row = min(rows - 1, y // row_spacing)          # Calculates in which grid for x value of that person 
        grid_col = min(cols - 1, x // col_spacing)          # Calculates in which grid for y value of that person 
        grid_key = f"{(grid_row * cols) + grid_col + 1}"
        grid_counts[grid_key] += 1                          # Increases by 1 for each person into that grid in dictionary.

    return grid_counts

def printGridCountsWithText(grid_counts, row_spacing, col_spacing, image):
    reference = 10   # amount of indentation from a grid`s top side
    positions = [   # This array is used to calculate where output to print is placed for only 3x3 grid map
    (0, reference), (col_spacing+2, reference), (2*col_spacing+2, reference),
    (0, row_spacing+reference), (col_spacing+2, row_spacing+reference), (2*col_spacing+2, row_spacing+reference),
    (0, 2*row_spacing+reference), (col_spacing+2, 2*row_spacing+reference), (2*col_spacing+2, 2*row_spacing+reference),]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.32                    # 0.5 for exact video, 0.3 for heatmap
    font_thickness = 1                  
    font_color = (0, 255, 0)            # Green

    # This for loop retieves people number from grid_counts dictionary and prints them accordingly.
    for grid_key, count in grid_counts.items():
        text = f"Grid{grid_key} = {count} people"
        cv2.putText(image, text, (positions[int(grid_key)-1][0],positions[int(grid_key)-1][1]), font, font_scale, font_color, font_thickness)
        if count > PEOPLE_THRESHOLD*AREA/9: alertGrid(image,(positions[int(grid_key)-1][0],positions[int(grid_key)-1][1]))

# Alert function to warn 
def alertGrid(image, position): 
    positionX, positionY = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.2                    
    font_thickness = 1                  
    font_color = (0, 0, 255)
    text = f"[ALERT] Too crowded area"
    cv2.putText(image, text, (positionX+10,positionY+25), font, font_scale, font_color, font_thickness) 

# Video processing loop starts here
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if the video has ended

    # Perform object detection
    results = model(frame)

    # Process the results to get coordinates
    people_coordinates = processResults(results , frame)
    person_points = np.array(people_coordinates, dtype='float32').reshape(-1, 1, 2)
    
    # Apply homography transformation
    if status.all():
        person_points_bird_view = cv2.perspectiveTransform(person_points, H)
        # Update the heatmap with bird's eye view points
        for point in person_points_bird_view:
            x, y = point.ravel()
            if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
                cv2.circle(heatmap, (int(x), int(y)), 3, max_intensity, -1)  # Set to max intensity for new points

        # Apply decay to the heatmap
        heatmap = np.clip(heatmap * decay_rate, 0, 300) ##sondaki 300 üsttekinin aynısı max instensity

        # Rotate and flip the heatmap
        heatmap_rotated = np.rot90(heatmap)
        heatmap_mirrored = np.flipud(heatmap_rotated)

        # Normalize the heatmap
        heatmap_blurred = cv2.GaussianBlur(heatmap_mirrored, (11, 11), 0)

        # Normalize the heatmap to the range [0, 255]
        heatmap_normalized = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX)

        # Apply the color map to generate a visual heatmap
        colored_heatmap = cv2.applyColorMap(np.uint8(heatmap_normalized), cv2.COLORMAP_JET)
        generateGrids(frame,colored_heatmap,people_coordinates)
        # Display the colored heatmap in a separate window
        cv2.imshow('Heatmap', colored_heatmap)

    # Display the video frame
    cv2.imshow('Frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()