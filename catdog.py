import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import cv2
import time
from PIL import Image

st.set_page_config(page_title='تشخیص سگ از گربه - RoboAi', layout='centered', page_icon='🐶')

model = load_model("cat_dog_model.h5")

def show_page():
    st.write("<h4 style='text-align: center; color: blue;'>تشخیص سگ از گربه 🐶</h4>", unsafe_allow_html=True)
    st.write("<h6 style='text-align: center; color: black;'>Robo-Ai.ir طراحی و توسعه</h6>", unsafe_allow_html=True)
    st.divider()
    st.link_button("Robo-Ai بازگشت به", "https://robo-ai.ir")

    container = st.container(border=True)
    container.write("<h6 style='text-align: right; color: gray;'>تشخیص تصویر سگ از گربه 🐱</h6>", unsafe_allow_html=True)

    image = st.file_uploader('آپلود تصویر', type=['jpg', 'jpeg', 'png'])
    
    if image:
        file_bytes = np.array(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels='BGR', use_container_width=True)

        if st.button('🔍 ارزیابی تصویر'): 
            label, confidence = predict_image(img)
            display_result(label, confidence)

def predict_image(img):
    img_resized = cv2.resize(img, (128, 128)) 
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  

    prediction = model.predict(img_array)[0][0]  
    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = round(max(prediction, 1 - prediction) * 100, 2)
    
    return label, confidence

def display_result(label, confidence):
    if label == "Dog":
        text1 = "✦ بر اساس ارزیابی من ، تصویر سگ رویت شد"
        text2 = "✦ اطمینان من از دقت محاسبه"

    else:
        text1 = "✦ بر اساس ارزیابی من ، تصویر گربه رویت شد"
        text2 = "✦ اطمینان من از دقت محاسبه"

    def stream_text(text):
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.09)
    
    st.write_stream(stream_text(text1))
    st.write_stream(stream_text(text2))
    st.write(f"**{confidence}%**")


show_page()
