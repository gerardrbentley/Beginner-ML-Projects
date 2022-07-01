from pathlib import Path
from random import randint

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="CV - Beginner Machine Learning",
    page_icon="🤖",
)
st.header("(CV) Computer Vision Demo 👀", "cv")

st.write(
    """'Object Detection' on any image.
The 'YOLOv4-tiny' model was trained to locate 80 different types of things!

Other awesome app features:
- Drag and Drop file upload
- Webcam Image processing

YOLO stands for "You Only Look Once"

Powered by [OpenCV](https://opencv.org/), [Pre-Trained YOLOv4-tiny Model](https://github.com/AlexeyAB/darknet), and [Streamlit](https://docs.streamlit.io/).
Built with ❤️ by [Gar's Bar](https://tech.gerardbentley.com/)
"""
)

# Initialize YOLO 'classes' to predict
class_names = Path("yolo/coco.names").read_text().split("\n")
# Make a random color for each class_name
colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in class_names]
# Load in the trained model
model = cv2.dnn_DetectionModel("yolo/yolov4-tiny.cfg", "yolo/yolov4-tiny.weights")
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

# Decide how the User will provide an image
use_upload = "Upload an Image"
use_camera = "Use Camera to take a Photo"
image_method = st.radio("How to select Image", [use_upload, use_camera])

if image_method == use_upload:
    image_file = st.file_uploader(
        "Upload Image File 🌄", ["png", "jpg", "jpeg"], accept_multiple_files=False
    )
elif image_method == use_camera:
    image_file = st.camera_input("Take a Photo 📸")

# Stop the app if the User hasn't chosen an image yet
if image_file is None:
    st.info(
        "Select an Image to proceed. Don't have one handy? How about [this one](https://github.com/AlexeyAB/darknet/raw/master/data/person.jpg) (right click and save link as)!"
    )
else:
    # Try to open the image with Python Imaging Library fork Pillow
    try:
        raw_image = Image.open(image_file)
    except Exception:
        st.error("Error: Invalid image. Try another one.")
        st.stop()

    # Convert the image to the right form for processing in the model
    image = cv2.resize(
        np.asarray(raw_image), dsize=(416, 416), interpolation=cv2.INTER_AREA
    )

    # Run the YOLO model on the image
    classes, scores, boxes = model.detect(image, 0.25, 0.5)
    for classid, score, box in zip(classes, scores, boxes):
        color = colors[classid]
        label = f"{class_names[classid]} : {score:.5f}"
        x, y, width, height = box
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

    st.image(image)

    with st.expander("Show Model Outputs"):
        st.write(list(zip(classes, scores, boxes)))

st.write(
    """## Take it further:

- Let users upload multiple images at once to analyze
- Train your own YOLO model to identify new objects
- Dive deeper into CV topics such as image classification, semantic segmentation, Optical Character Recognition (OCR), etc.
- Extend previous techniques to video
"""
)


if st.checkbox("Show All 80 'Classes' that YOLO can predict"):
    st.write(class_names)

if st.checkbox("Show Code (~90 lines)"):
    with open(__file__, "r") as f:
        st.code(f.read())
