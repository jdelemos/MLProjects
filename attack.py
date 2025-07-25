import pyautogui
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import time
from inference import get_model
import supervision as sv
import pyautogui
import cv2
import os
from inference_sdk import InferenceHTTPClient
import json
from pprint import pprint


# Load your trained YOLOv8 model
# model = YOLO("/home/jonathon-delemos/Desktop/Gaming with ML/osrs_bot/best.pt")

# Load the model you trained on Roboflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com", api_key="axF6fjbxGk5aiJmkAGPq"
)


# Define screen region to capture: (left, top, width, height)
region = (1124, 28, 796, 510)
screenshot_count = 0

# Create folder to save images
save_dir = "/home/jonathon-delemos/Desktop/Gaming with ML/osrs_bot/training_data/"

while True:
    # Step 1: Capture a screenshot of the defined region
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)
    screenshot_count += 1

    #  # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Save to disk temporarily (Roboflow API needs file input)
    temp_path = os.path.join(save_dir, f"temp_frame{screenshot_count}.jpg")
    cv2.imwrite(temp_path, frame_bgr)

    # Step 3: Infer the probability based off the image
    curr_time = time.time()
    response = client.run_workflow(
        workspace_name="imagerecognition-nrs3b",
        workflow_id="small-object-detection-sahi-2",
        images={"image": temp_path},
        use_cache=True,  # cache workflow definition for 15 minutes
    )
    # pprint(response)
    nex_time = time.time()
    print(nex_time - curr_time)
    # 3. Draw boxes
    for pred in response[0]["predictions"]["predictions"]:
        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])
        g = float(pred["confidence"])
        class_name = pred["class"]

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = y + h // 2

        print(x1, x2, y1, y2)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            class_name,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )
        print("saving file..")
        # 5. Save annotated image
        filename = f"screenshot_{screenshot_count}.png"
        cv2.imwrite(os.path.join(save_dir, filename), frame_bgr)

        if class_name == "cow" and g > 0.60:
            # 4. Move mouse & click (center of bounding box)
            click_x = region[0] + x
            click_y = region[1] + y
            print(f"Clicking at: ({click_x}, {click_y}) on {class_name}")
            pyautogui.moveTo(click_x, click_y, duration=0.5)
            pyautogui.click()
            time.sleep(10)
            break
