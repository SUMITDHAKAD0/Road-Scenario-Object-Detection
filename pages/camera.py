
import streamlit as st
import cv2
import os
import time
from ultralytics import YOLO
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

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

# load model
model = YOLO(model_path)
threshold = 30
class_name_dict = {0: 'bike', 1:'car', 2:'green_traffic_signs', 3:'person', 4:'traffic_signal', 5:'traffic_signs'}
class_color_dict = {0: (6,6,255), 1:(255,6,106), 2:(72,255,6), 3:(0,255,222), 4:(0,68,255), 5:(0,0,255)}


cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
stop_button_pressed = st.button("Stop")
while cap.isOpened() and not stop_button_pressed:
    ret, frame = cap.read()
   
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
    if not ret:
        st.write("Video Capture Ended")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame,channels="RGB")

    if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
        break
cap.release()
cv2.destroyAllWindows()
