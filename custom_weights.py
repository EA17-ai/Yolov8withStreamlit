import time
import cv2
import streamlit as st
from pytube import YouTube
from PIL import Image
from torchvision import io
from ultralytics import YOLO

# Function to load the YOLO model
@st.cache_resource
def load_model(path):
    try:
        model = YOLO(path)
    except Exception as ex:
        st.error(f"Error loading the model: {ex}")
        st.stop()

    return model

# List of class names for object detection
classNames = []  #add class names according to your requirement

# Load the YOLO model
model_path = "yolov8n.pt" #add your custom weights
model = load_model(model_path)

# Set up the Streamlit layout and options
option = st.sidebar.selectbox(
    'Which one would you like to use?',
    ('Photo', 'Video', "Video Link", 'Webcam'))

# Option for processing a photo
if option == "Photo":
    # Object detection options
    objects_list = [classNames.index(option) for option in st.sidebar.multiselect(
        'Pick Objects', options=classNames)]
    conf = float(st.sidebar.slider("Select Confidence Level", 30, 100, 50)) / 100

    # Upload an image
    uploadedFile = st.sidebar.file_uploader("Upload Your Image", type=["png", "jpg", "jpeg"])
    tab1, tab2 = st.tabs(["Original", "Predicted"])

    if uploadedFile is not None:
        try:
            # Read the uploaded image
            img = Image.open(uploadedFile)
            img.save("images/img.jpg")
            image = cv2.cvtColor(cv2.imread("images/img.jpg"), cv2.COLOR_BGR2RGB)

            # Perform object detection on the image
            results = model.predict(image, conf=conf, classes=objects_list)
            res_plotted = results[0].plot()

            # Display the original and predicted images
            with tab1:
                st.header("Original Image")
                st.image(image, width=200, use_column_width=True)
            with tab2:
                st.header("Predicted Image")
                st.image(res_plotted, channels="RGB", use_column_width=True, width=300)

        except Exception as ex:
            st.error(f"Error processing the image: {ex}")

    else:
        st.write("Upload an image for detection")

# Option for processing a video from a link
if option == "Video Link":
    # Object detection options
    link = st.text_input("Please Enter YouTube link")
    objects_list = [classNames.index(option) for option in st.sidebar.multiselect(
        'Pick Objects', options=classNames)]
    conf = float(st.sidebar.slider("Select Confidence Level", 30, 100, 50)) / 100

    stframe = st.empty()
    if st.sidebar.button("Detect"):
        try:
            # Fetch the video from YouTube
            youtube = YouTube(link)
            stream = youtube.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

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

        except Exception as ex:
            st.error(f"Error processing the video link: {ex}")

# Option for processing a video
if option == "Video":
    # Object detection options
    objects_list = [classNames.index(option) for option in st.sidebar.multiselect(
        'Pick Objects', options=classNames)]
    conf = float(st.sidebar.slider("Select Confidence Level", 30, 100, 50)) / 100

    # Upload a video file
    uploadedFile = st.sidebar.file_uploader("Upload Your Video", type=["mp4", "avi"])

    if uploadedFile is not None:
        try:
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

        except Exception as ex:
            st.error(f"Error processing the video: {ex}")

# Option for using the webcam
if option == "Webcam":
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
        vid_cap.set(3, 640)
        vid_cap.set(4, 480)
    except Exception as ex:
        st.error(f"Error accessing the webcam: {ex}")

    if "running" not in st.session_state:
        st.session_state.running = False

    if detect_objects:
        if not st.session_state.running:
            st.session_state.running = True
            stframe = col1.empty()

            while vid_cap.isOpened() and st.session_state.running:
                success, image = vid_cap.read()

                try:
                    # Perform object detection on the video frame
                    results = model.predict(image, conf=conf, classes=objects_list)
                    res_plotted = results[0].plot()

                    # Display detected objects
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            with placeholder.container():
                                x1, y1, x2, y2 = box.xyxy[0]
                                cls = int(box.cls[0])
                                placeholder.write(f"{classNames[cls]},{int(x1)},{int(y1)},{int(x2)},{int(y2)}")

                    stframe.image(res_plotted,
                                  caption='Detected Video',
                                  channels="BGR",
                                  use_column_width=True)

                except Exception as ex:
                    st.error(f"Error processing webcam frame: {ex}")
