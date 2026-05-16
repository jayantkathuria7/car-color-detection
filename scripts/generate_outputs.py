import os
import cv2
from utils.pipeline import handle_image, process_video 

BASE_DIR = os.getcwd()
input_folder = os.path.join(BASE_DIR,"assets/test/input/")
output_folder = os.path.join(BASE_DIR,"assets/test/output/")
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    # --- Images ---
    if filename.lower().endswith((".jpg", ".png")):
        input_img = cv2.imread(file_path)
        annotated_img, counts = handle_image(input_img)
        output_image = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        out_path = os.path.join(output_folder, filename.replace(".", "_out."))
        cv2.imwrite(out_path, output_image)
        print(f"[IMAGE] Saved annotated: {out_path}")

    # --- Videos ---
    elif filename.lower().endswith(".mp4"):
        out_path = os.path.join(output_folder, filename.replace(".mp4", "_out.mp4"))
        output_video_path = process_video(file_path, out_path)
        print(f"[VIDEO] Saved annotated: {out_path}")

print("✅ All files processed!")