import streamlit as st
import os
import time
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.title("WelCome To Road Scenerio Detection")
# st.header("Objects:-Car,Bike,Person,Traffic_lights,Traffic_signs,green_traffic_signs")
text = '''
    To detect  object you can use WebCam, Image and video\n
    We Support Objects Classes like:\n
        0: bike\n
        1: car\n
        2: green_traffic_board\n
        3: person\n
        4: traffic_signal\n
        5: traffic_signs                
'''
st.markdown(text)


uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg"])
if uploaded_file != None:
    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
    save_folder = "results"
    os.makedirs(save_folder, exist_ok=True)
      
    vid = uploaded_file.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_file.read()) # save video to disk
    output_path = os.path.join(save_folder, uploaded_file.name)

    # load model
    model = YOLO(model_path)
    threshold = 30
    class_name_dict = {0: 'bike', 1:'car', 2:'green_traffic_signs', 3:'person', 4:'traffic_signal', 5:'traffic_signs'}
    class_color_dict = {0: (6,6,255), 1:(255,6,106), 2:(72,255,6), 3:(0,255,222), 4:(0,68,255), 5:(0,0,255)}

    # st.balloons()
    submit_button=st.button("submit")
    if submit_button:
        st.success("Video submitted! 🎉🎈🥳")
        cap = cv2.VideoCapture(vid)
        ret, frame = cap.read()
        H, W, _ = frame.shape
        

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

        while ret:

            results = model(frame)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                score = round(score*100, 2)
                if score > threshold:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), class_color_dict[int(class_id)], 4)
                    cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, class_color_dict[int(class_id)], 3, cv2.LINE_AA)
                    cv2.putText(frame, str(score)+"%", (int(x1), int(y1 - 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, class_color_dict[int(class_id)], 3, cv2.LINE_AA)
            out.write(frame)
            ret, frame = cap.read()
        # st.video(out)
            

        cap.release()
        out.release()
        cv2.destroyAllWindows()
