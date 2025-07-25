import pyautogui
import time
import os

# Create folder to save images
save_dir = "/home/jonathon-delemos/Desktop/Gaming with ML/osrs_bot/training_data/"

screenshot_count = 0
interval = 3  # seconds

print("Starting automated screenshot capture... Press Ctrl+C to stop.")

try:
    while True:
        region = (1124, 28, 796, 510)  # Modify this based on your game window
        # Replace with your game window coordinates
        screenshot = pyautogui.screenshot(region=region)
        filename = os.path.join(save_dir, f"screenshot_{screenshot_count}.png")
        screenshot.save(filename)
        print(f"Saved: {filename}")

        screenshot_count += 1
        time.sleep(interval)
        if screenshot_count > 75:
            break
except KeyboardInterrupt:
    print("Stopped.")
