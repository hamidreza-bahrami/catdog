import streamlit as st
import numpy as np
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from PIL import Image
import cv2
import time

model = load_model('model.h5')

def show_page():
    st.write("<h3 style='text-align: center; color: blue;'>تشخیص سگ از گربه 🐶</h3>", unsafe_allow_html=True)
    st.write("<h6 style='text-align: center; color: black;'>Robo-Ai.ir طراحی و توسعه</h6>", unsafe_allow_html=True)
    st.link_button("Robo-Ai بازگشت به", "https://robo-ai.ir")
    container = st.container(border=True)
    container.write("<h6 style='text-align: right; color: gray;'>تشخیص تصویر سگ از گربه 🐱</h6>", unsafe_allow_html=True)
    st.write('')

    with st.sidebar:
        st.write("<h5 style='text-align: center; color: blcak;'>تشخیص نژاد های مختلف سگ از گربه</h5>", unsafe_allow_html=True)
        st.divider()
        st.write("<h5 style='text-align: center; color: black;'>طراحی و توسعه</h5>", unsafe_allow_html=True)
        st.write("<h5 style='text-align: center; color: gray;'>حمیدرضا بهرامی</h5>", unsafe_allow_html=True)

    image = st.file_uploader('آپلود تصویر', type=['jpg', 'jpeg'])
    button = st.button('ارزیابی تصویر')       
    if image is not None:
        file_bytes = np.array(bytearray(image.read()), dtype= np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels= 'BGR', use_container_width= True)
        if button: 
            x = cv2.resize(img, (128, 128))
            x1 = img_to_array(x)
            x1 = x1.reshape((1,) + x1.shape)
            # y_pred = model.predict(x1)
            prediction = model.predict(x1)[0][0]
            label = 1 if prediction > 0.5 else 0
            confidence = prediction if prediction > 0.5 else 1 - prediction
            if label == 1:
                text1 = 'بر اساس ارزیابی من ، تصویر سگ رویت شد'
                text2 = 'Based on my analysis, Dog was seen in this image'
                text3 = 'اطمینان من از دقت محاسبه'
                text4 = (confidence)
                def stream_data1():
                    for word in text1.split(" "):
                        yield word + " "
                        time.sleep(0.09)
                st.write_stream(stream_data1)
                def stream_data2():
                    for word in text2.split(" "):
                        yield word + " "
                        time.sleep(0.09)
                st.write_stream(stream_data2)
                def stream_data3():
                    for word in text3.split(" "):
                        yield word + " "
                        time.sleep(0.09)
                st.write_stream(stream_data3)
                st.markdown(text4)

            elif label == 0:
                text4 = 'بر اساس ارزیابی من ، تصویر گربه رویت شد'
                text5 = 'Based on my analysis, Cat was seen in this image'
                text6 = 'اطمینان من از دقت محاسبه'
                text7 = (confidence)
                def stream_data4():
                    for word in text4.split(" "):
                        yield word + " "
                        time.sleep(0.09)
                st.write_stream(stream_data4)
                def stream_data5():
                    for word in text5.split(" "):
                        yield word + " "
                        time.sleep(0.09)
                st.write_stream(stream_data5)
                def stream_data6():
                    for word in text6.split(" "):
                        yield word + " "
                        time.sleep(0.09)
                st.write_stream(stream_data6)
                st.markdown(text7)

show_page()
