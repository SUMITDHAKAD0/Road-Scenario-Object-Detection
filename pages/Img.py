
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

input_img = st.file_uploader("Choose the Picture")
if input_img != None:
    image = Image.open(input_img)
    st.image(image)   
    input_img = np.array(image) 

    #model loading
    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
    model = YOLO(model_path)

    # parameters
    threshold = 30
    class_name_dict = {0: 'bike', 1:'car', 2:'green_traffic_signs', 3:'person', 4:'traffic_signal', 5:'traffic_signs'}
    class_color_dict = {0: (6,6,255), 1:(255,6,106), 2:(72,255,6), 3:(0,255,222), 4:(0,68,255), 5:(0,0,255)}

    if st.button('Submit'):
        results = model(input_img)[0]
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            score = round(score*100, 2)
            if score > threshold:
                cv2.rectangle(input_img, (int(x1), int(y1)), (int(x2), int(y2)), class_color_dict[int(class_id)], 4)
                cv2.putText(input_img, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, class_color_dict[int(class_id)], 3, cv2.LINE_AA)
                cv2.putText(input_img, str(score)+"%", (int(x1), int(y1 - 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, class_color_dict[int(class_id)], 3, cv2.LINE_AA)
        cv2.imwrite('result/output.png', input_img)
        cv2.destroyAllWindows()
        st.image(input_img)
        st.balloons()