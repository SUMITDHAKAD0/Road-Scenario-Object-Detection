import streamlit as st
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

image = Image.open('Capture.jpg')
st.image(image, caption='Sunrise by the mountains')
