import pyautogui
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import time
import supervision as sv
import os
import random

import json
from pprint import pprint


# Load your trained YOLOv8 model
# model = YOLO("/home/jonathon-delemos/Desktop/Gaming with ML/osrs_bot/best.pt")


# Define screen region to capture: (left, top, width, height)
region = (1124, 28, 796, 510)
screenshot_count = 0

photo_dir = "/home/jonathon-delemos/Desktop/Gaming with ML/osrs_bot/photos/Health/"

# Grab all .png files in that folder
health_images = [
    os.path.join(photo_dir, f) for f in os.listdir(photo_dir) if f.endswith(".png")
]


# Load your trained model (e.g., from Roboflow export or your own training)
model = YOLO(
    "/home/jonathon-delemos/Desktop/Gaming with ML/osrs_bot/best.pt"
)  # or yolov8n.pt, yolov8s.pt, etc.


# Create folder to save images
save_dir = "/home/jonathon-delemos/Desktop/Gaming with ML/osrs_bot/training_data/"

while True:

    # This is my stupid solution to image not recognized on screen. Also, it's taking awhile to determine if I need health fill.
    # this simply won't work. The screen reading is taking more than half a second which allows the target to move.
    # try:
    #     # check health
    #     image = random.choice(health_images)
    #     location = pyautogui.locateCenterOnScreen(image, confidence=0.8)
    #     if location:
    #         x, y = location
    #         pyautogui.moveTo(x, y, duration=0.0)
    # except:
    #     print("Health either not low enough or image not found")

    # Step 1: Capture a screenshot of the defined region
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)
    screenshot_count += 1

    #  # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Save to disk temporarily (Roboflow API needs file input)
    temp_path = os.path.join(save_dir, f"temp_frame{screenshot_count}.jpg")
    cv2.imwrite(temp_path, frame_bgr)

    # Run detection on a single image
    results = model(
        f"/home/jonathon-delemos/Desktop/Gaming with ML/osrs_bot/training_data/temp_frame{screenshot_count}.jpg",
        save=True,
        conf=0.25,
    )

    boxes = results[0].boxes
    # dictionary that stores {0:cows, 1:map..}
    names = results[0].names

    # 3. Check to see if the boxes array is empty
    if boxes is not None:
        # loop through the boxes object to get all detections: box.xyxy box.confidence
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Get confidence, for some reason it's box.conf[0].item?
            conf = box.conf[0].item()

            # Get class name using box.cls[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id]

            # Get center
            cx = int((x1 + x2 - 5) / 2)
            cy = int((y1 + y2 - 5) / 2)

            cv2.rectangle(
                frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            )
            cv2.putText(
                frame_bgr,
                cls_name,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        print("saving file..")
        # 5. Save annotated image
        filename = f"screenshot_{screenshot_count}.png"
        cv2.imwrite(os.path.join(save_dir, filename), frame_bgr)

        if cls_name == "cow" and conf > 0.25:
            # 4. Move mouse & click (center of bounding box)
            click_x = region[0] + cx
            click_y = region[1] + cy
            print(f"Clicking at: ({click_x}, {click_y}) on {cls_name}")
            pyautogui.moveTo(click_x, click_y, duration=0.0)
            pyautogui.click()
            pyautogui.keyDown("left")
            zero_thru_one = random.random()
            time.sleep(zero_thru_one)
            pyautogui.keyUp("left")
            time.sleep(10)
