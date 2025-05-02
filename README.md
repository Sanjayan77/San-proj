# San-proj
#pathole detection 
# ===============================
# Pothole Detection and Growth Analysis using YOLO + Paris' Law
# Author: Sanjayan
# Purpose: Detect potholes in road images, classify severity, predict future growth using engineering fracture mechanics
# ===============================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
import os
from datetime import datetime

# ===============================
# 1. Load YOLOv8 model
# ===============================
print("[INFO] Loading YOLO model...")
model = YOLO("yolov8n.pt")  # Replace with fine-tuned model for pothole detection if available

# ===============================
# 2. Load Road Image
# ===============================
image_path = "road.jpg"  # Replace with path to your drone image
if not os.path.exists(image_path):
    raise FileNotFoundError("❌ Image file not found. Please check the path.")

image = cv2.imread(image_path)
original_image = image.copy()
print(f"[INFO] Image loaded with shape: {image.shape}")

# ===============================
# 3. Run Inference
# ===============================
print("[INFO] Running detection...")
results = model(image)

# ===============================
# 4. Define Classification Logic
# ===============================
def classify_pothole_by_area(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    if area < 1000:
        return "Minor", area
    elif area < 3000:
        return "Moderate", area
    else:
        return "Severe", area

# ===============================
# 5. Draw Boxes and Store Info
# ===============================
detections = []
crack_areas = []

for r in results:
    for box in r.boxes.xyxy:
        box = box.int().cpu().numpy()
        x1, y1, x2, y2 = box
        severity, area = classify_pothole_by_area((x1, y1, x2, y2))
        crack_areas.append(area)

        # Draw rectangle and label
        color = (0, 255, 0) if severity == "Minor" else (0, 255, 255) if severity == "Moderate" else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{severity}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Store data
        detections.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "severity": severity,
            "area": area,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

print(f"[INFO] {len(detections)} potholes detected and classified.")

# ===============================
# 6. Save Detection Results to CSV
# ===============================
df = pd.DataFrame(detections)
csv_name = "pothole_detections.csv"
df.to_csv(csv_name, index=False)
print(f"[INFO] Detections saved to {csv_name}")

# ===============================
# 7. Show Detection Image
# ===============================
cv2.imshow("Pothole Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===============================
# 8. Paris' Law - Crack Growth Simulation
# ===============================

def simulate_paris_law(a0, delta_K, C=1e-11, m=3, cycles=10000):
    """
    Simulate crack growth over time based on Paris' Law.
    """
    a = [a0]
    for i in range(cycles):
        da = C * (delta_K ** m)
        a.append(a[-1] + da)
    return a

# Simulate crack growth for each detected pothole
print("[INFO] Simulating pothole growth using Paris' Law...")
plt.figure(figsize=(10, 6))

for idx, area in enumerate(crack_areas):
    # Normalize area into initial crack size estimate
    a0 = area / 10000  # arbitrary scaling for visualization
    delta_K = 1.5 + (idx % 3) * 0.4  # vary for realism
    growth = simulate_paris_law(a0, delta_K)
    plt.plot(growth, label=f'Pothole {idx+1} ({df.iloc[idx]["severity"]})')

plt.title("📈 Predicted Pothole Growth Over Time (Paris’ Law)")
plt.xlabel("Load Cycles (e.g., vehicles passing over)")
plt.ylabel("Estimated Crack Length (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pothole_growth_prediction.png")
plt.show()
print("[INFO] Growth predictions visualized and saved.")

# ===============================
# 9. Smart City Output Summary
# ===============================
print("\n🚧 Smart Maintenance Recommendation:")
for row in detections:
    if row['severity'] == "Severe":
        print(f"➡️ ALERT: Severe pothole at ({row['x1']},{row['y1']}) — schedule immediate repair!")
    elif row['severity'] == "Moderate":
        print(f"🔶 Notice: Moderate pothole at ({row['x1']},{row['y1']}) — inspect within 3 days.")
    else:
        print(f"✅ Minor pothole at ({row['x1']},{row['y1']}) — log for monitoring.")

print("\n✅ All systems completed.")
