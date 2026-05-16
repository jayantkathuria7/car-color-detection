from moviepy import VideoFileClip
import os

BASE_DIR = os.getcwd()
INPUT_PATH = os.path.join(BASE_DIR, "assets/test/output/sample_video2_out.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "assets/demo.gif")
print(OUTPUT_PATH,INPUT_PATH)
clip = VideoFileClip(INPUT_PATH)
clip = clip.subclipped(3,7)  # first 10 seconds
clip.write_gif(OUTPUT_PATH)