import cv2

input_video = r"C:\Users\LENOVO\Desktop\spermchannelvid.mp4"
# Correct path, raw string
output_video = "cropped_video_3min.avi"  # Used .avi for XVID codec was having challenges with .mp4

# Crop coords (x, y, w, h)
x, y, w, h = 2558, 1466, 1240, 446

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)
if fps == 0:
    raise Exception("FPS is zero. Check if input video exists and is not corrupt!")

fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Used 'XVID' for .avi
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))  # (width, height)

max_frames = int(fps * 3 * 60)  # 3 minutes
count = 0

while count < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed or end of video.")
        break
    crop = frame[y : y + h, x : x + w]
    # Sanity check crop size
    if crop.shape[0] != h or crop.shape[1] != w:
        print(f"Crop shape mismatch at frame {count}: {crop.shape}")
        break
    out.write(crop)
    count += 1

cap.release()
out.release()
print(f"Cropped video of first 3 minutes saved as {output_video}")
