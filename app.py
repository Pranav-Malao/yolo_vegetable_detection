import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import sys
import cv2
import numpy as np


st.title("🥗 Vegetable Detection App (YOLO)")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) # Convert PIL RGB to OpenCV BGR

    # --- User-defined paths ---
    MODEL_PATH = './yolo_models_testing/final_11s.pt'  # Replace with the actual path to your YOLO model file
    MIN_CONF_THRESHOLD = 0.5
    DISPLAY_RESOLUTION = '640x640'  # Set to '640x480' or similar if you want to resize the display
    SAVE_ANNOTATED_IMAGE = True
    OUTPUT_IMAGE_PATH = os.path.join("./yolo_models_testing/annotated_images", f"annotated_{uploaded_file.name}")
    # --------------------------

    # Check if model file exists and is valid
    if not os.path.exists(MODEL_PATH):
        st.error('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
        sys.exit(0)

    # Load the model into memory and get labemap
    try:
        model = YOLO(MODEL_PATH, task='detect')
        labels = model.names
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        sys.exit(0)

    # Parse user-specified display resolution
    resize = False
    if DISPLAY_RESOLUTION:
        resize = True
        try:
            resW, resH = int(DISPLAY_RESOLUTION.split('x')[0]), int(DISPLAY_RESOLUTION.split('x')[1])
        except ValueError:
            st.error("ERROR: Invalid display resolution format. Please use 'WxH' (e.g., '640x480').")
            sys.exit(0)

    # Set bounding box colors (using the Tableu 10 color scheme)
    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
                   (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

    # Resize frame to desired display resolution
    if resize:
        resized_frame = cv2.resize(frame, (resW, resH))
    else:
        resized_frame = frame

    # Run inference on frame
    results = model(resized_frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    object_count = 0
    annotated_frame = resized_frame.copy() # Create a copy to draw on

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):

        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > MIN_CONF_THRESHOLD:

            color = bbox_colors[classidx % 10]
            cv2.rectangle(annotated_frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(annotated_frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
            cv2.putText(annotated_frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

            # Basic example: count the number of objects in the image
            object_count += 1

    # Display detection results
    cv2.putText(annotated_frame, f'Number of objects: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Draw total number of detected objects

    # Convert the annotated frame back to RGB for Streamlit display
    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Detected Vegetables", use_container_width=True)

    if SAVE_ANNOTATED_IMAGE:
        try:
            cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_frame)
            st.success(f"Annotated image saved as '{OUTPUT_IMAGE_PATH}'")
        except Exception as e:
            st.error(f"Error saving the annotated image: {e}")

    # You don't need to manually show the OpenCV window in Streamlit
    # cv2.imshow('YOLO detection results', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()