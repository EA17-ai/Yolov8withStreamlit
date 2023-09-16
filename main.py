import time
import numpy as np
from torchvision import io
from ultralytics import YOLO
from PIL import Image
import streamlit as st
import cv2
from pytube import YouTube


# Function to load the YOLO model
@st.cache_resource
def load_model(path):
    try:
        model = YOLO(path)
    except Exception as ex:
        print(ex)
        st.write(f"Unable to load model. Check the specified path: {model_path}")

    return model


# List of class names for object detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load the YOLO model
model_path = "yolov8n.pt"
model = load_model(model_path)

# Set up the Streamlit layout and options
option = st.sidebar.selectbox(
    'Which one would you like to use?',
    ('Photo', 'Video', "Video Link", 'Webcam'))


# Option for processing a photo
if option == "Photo":
    objects_list = [classNames.index(option) for option in st.sidebar.multiselect(
        'Pick Objects', options=classNames)]

    conf = float(st.sidebar.slider("Select Confidence Level", 30, 100, 50)) / 100
    #column1, column2 = st.columns(2)
    uploadedFile = st.sidebar.file_uploader("Upload Your Image", type=["png", "jpg", "jpeg"])
    tab1, tab2= st.tabs(["Original", "Predicted"])
    if uploadedFile is not None:
        # Read the uploaded image
        img = Image.open(uploadedFile)
        img.save("images/img.jpg")
        image = cv2.cvtColor(cv2.imread("images/img.jpg"), cv2.COLOR_BGR2RGB)

        # Perform object detection on the image
        results = model.predict(image, conf=conf, classes=objects_list)
        res_plotted = results[0].plot()
        with tab1:
            st.header("Original Image")
            st.image(image, width=200,use_column_width=True)
        with tab2:
            st.header("Predicted Image")
            st.image(res_plotted, channels="RGB",use_column_width=True,width=300)
        #column1.image(,use_column_width=True)

        #column2.image(res_plotted, channels="RGB",use_column_width=True)
    else:
        st.write("Upload an image for detection")
if option == "Video Link":
    link=st.text_input("Please Enter youtube link")
    objects_list = [classNames.index(option) for option in st.sidebar.multiselect(
        'Pick Objects', options=classNames)]

    conf = float(st.sidebar.slider("Select Confidence Level", 30, 100, 50)) / 100

    stframe = st.empty()
    if st.sidebar.button("Detect"):
        youtube = YouTube(link)
        stream = youtube.streams.filter(file_extension="mp4", res=720).first()
        vid_cap = cv2.VideoCapture(stream.url)
        stframe = st.empty()
        while vid_cap.isOpened():
            success, frame = vid_cap.read()
            if not success:
                break
            else:
            # Perform object detection on the video frame
                results = model.predict(frame, conf=conf, classes=objects_list)
                res_plotted = results[0].plot()
                stframe.image(res_plotted,
                          caption='Detected Video',
                          channels="BGR",
                          use_column_width=True)

        vid_cap.release()

# Option for processing a video
if option == "Video":
    objects_list = [classNames.index(option) for option in st.sidebar.multiselect(
        'Pick Objects', options=classNames)]
    conf = float(st.sidebar.slider("Select Confidence Level", 30, 100, 50)) / 100

    uploadedFile = st.sidebar.file_uploader("Upload Your Video", type=["mp4", "avi"])
    # Option for processing a video
    if uploadedFile is not None:
        # Read the uploaded video file
        video_file = io.BytesIO(uploadedFile.read())
        vid_cap = cv2.VideoCapture(video_file)
        st.image(Image.open("images/wb.jpg"))
        stframe = st.empty()
        while vid_cap.isOpened():
            success, frame = vid_cap.read()

            if not success:
                break
            else:
            # Perform object detection on the video frame
                results = model.predict(frame)
                res_plotted = results[0].plot()

                stframe.image(res_plotted,
                          caption='Detected Video',
                          channels="BGR",
                          use_column_width=True)

        vid_cap.release()

# Option for using the webcam
if option == "Webcam":
    # st.image(Image.open("images/wb.jpg").resize((600, 400)))

    col1, col2 = st.columns([6, 1])
    placeholder = col2.empty()

    st.sidebar.title("Options")
    selected_camera = st.sidebar.radio("Available Cameras", [0, 1, 2, 3])
    objects_list = [classNames.index(option) for option in st.sidebar.multiselect(
        'Pick Objects', options=classNames)]

    conf = float(st.sidebar.slider("Select Confidence Level", 30, 100, 50)) / 100
    detect_objects = st.sidebar.button('Detect Objects')
    stop = st.sidebar.button("STOP")

    try:
        vid_cap = cv2.VideoCapture(selected_camera)
    except:
        col1.write("No Camera Available")

    vid_cap.set(3, 640)
    vid_cap.set(4, 480)

    if "running" not in st.session_state:
        st.session_state.running = False

    if detect_objects:
        if not st.session_state.running:
            st.session_state.running = True
            stframe = col1.empty()

            while vid_cap.isOpened() and st.session_state.running:
                success, image = vid_cap.read()

                # Perform object detection on the video frame
                results = model.predict(image, conf=conf, classes=objects_list)
                res_plotted = results[0].plot()
                stframe.image(res_plotted,
                              caption='Detected Video',
                              channels="BGR",
                              use_column_width=True)

                time.sleep(1)
                placeholder.empty()

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        with placeholder.container():
                            x1, y1, x2, y2 = box.xyxy[0]
                            cls = int(box.cls[0])
                            placeholder.write(f"{classNames[cls]},{int(x1)},{int(y1)},{int(x2)},{int(y2)}")

    if stop:
        st.session_state.running = False
        col1.empty()
        placeholder.empty()
        vid_cap.release()
        cv2.destroyAllWindows()
