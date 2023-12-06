import cv2
from ultralytics import YOLO
import os
import torch 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device ="0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0)
print(device)

# Load the YOLOv8 model
seg_model = YOLO('yolov8n-seg.pt')
seg_model.classes = ['person']
pose_model = YOLO('yolov8n-pose.pt')

# Open the video file
video_path = "gait.mp4"
cap = cv2.VideoCapture(video_path)
count = 0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame = cv2.resize(frame,(500, 480), interpolation = cv2.INTER_AREA)
    
    if success:
        #inferencing for person segmentation
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = seg_model.track(frame, persist=True) 
        if results[0].masks is None:
            continue
        
        mask = (results[0].masks.data[0].cpu().numpy() * 255).astype('uint8')
        mask = cv2.resize(mask,(500, 480), interpolation = cv2.INTER_AREA)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        pose_results = pose_model(frame)
        if pose_results[0].boxes is None:
            continue
        # Visualize the results on the frame
        annotated_frame_pose = pose_results[0].plot()

        #cv2.imwrite(str(f'{seg_class}s.jpg'), result)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        cv2.imshow("YOLOv8 Pose Tracking", annotated_frame_pose)
        cv2.imshow('Background remove', mask)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
