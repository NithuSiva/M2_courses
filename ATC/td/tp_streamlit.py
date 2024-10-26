import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2

st.title('MNIST APP')

canvas_result = st_canvas(
    height=300,
    width=300,
    background_color="rgba(255, 255, 255, 0.3)",
    key="canvas",
)

if st.button("Valider"):
    gray_image = gray = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_image, (28, 28), interpolation= cv2.INTER_LINEAR)

    model = tf.keras.models.load_model("cnn_model.keras")
    prediction = model.predict(resized_img.reshape(1,28,28,1))
    st.write(np.argmax(prediction))