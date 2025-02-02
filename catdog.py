import streamlit as st
import numpy as np
import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from PIL import Image
import cv2
import time

model = load_model('model.h5')

def classify_image(img):
    x = cv2.resize(img, (128, 128))
    x1 = img_to_array(x)
    x1 = x1.reshape((1,) + x1.shape)
    prediction = model.predict(x1)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

def show_page():
    st.write("<h3 style='text-align: center; color: blue;'>تشخیص سگ از گربه 🐶</h3>", unsafe_allow_html=True)
    st.write("<h6 style='text-align: center; color: black;'>Robo-Ai.ir طراحی و توسعه</h6>", unsafe_allow_html=True)
    st.link_button("Robo-Ai بازگشت به", "https://robo-ai.ir")
    container = st.container(border=True)
    container.write("<h6 style='text-align: right; color: gray;'>تشخیص تصویر سگ از گربه 🐱</h6>", unsafe_allow_html=True)
    st.write('')

    with st.sidebar:
        st.write("<h5 style='text-align: center; color: black;'>تشخیص نژاد های مختلف سگ از گربه</h5>", unsafe_allow_html=True)
        st.divider()
        st.write("<h5 style='text-align: center; color: black;'>طراحی و توسعه</h5>", unsafe_allow_html=True)
        st.write("<h5 style='text-align: center; color: gray;'>حمیدرضا بهرامی</h5>", unsafe_allow_html=True)

    image = st.file_uploader('آپلود تصویر', type=['jpg', 'jpeg'])
    button = st.button('ارزیابی تصویر')       
    
    if image is not None:
        file_bytes = np.array(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels='BGR', use_container_width=True)
        
        if button:
            label, confidence = classify_image(img)
            text1 = f'بر اساس ارزیابی من ، تصویر {"سگ" if label == "Dog" else "گربه"} رویت شد'
            text2 = f'Based on my analysis, {label} was seen in this image'
            text3 = 'اطمینان من از دقت محاسبه'
            text4 = f'{confidence:.2f}'
            
            def stream_data(text):
                for word in text.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            
            st.write_stream(stream_data(text1))
            st.write_stream(stream_data(text2))
            st.write_stream(stream_data(text3))
            st.markdown(text4)

show_page()
