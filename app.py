import tensorflow as tf
from tensorflow import keras
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2

classes = ["Biodegradable Waste","Non Biodegradable Waste"]
model = keras.models.load_model("B_Or_N.h5")

def prediction(uploaded_file,model):
    size = (180, 180)
    image = ImageOps.fit(uploaded_file, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img_resize = cv2.resize(image, dsize=(224, 224))
    img_reshape = tf.expand_dims(img_resize,axis=0)
    predictions = model.predict(img_reshape)
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    return predictions

html_temp = """
    <div style ="background-color:orange;padding:13px">
    <h1 style ="color:black;text-align:center;">Garbage Biodegradable Detection ML App </h1>
    </div>
    """

# this line allows us to display the front end aspects we have
# defined in the above code
st.markdown(html_temp, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file",type=['jpg','png','jpeg'])

if uploaded_file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    predictions = prediction(image, model)
    ans = classes[predictions.numpy()[0][0]]
    if ans == 0:
        st.success('The image is classified as {}'.format(ans))
    else:
        st.success('The image is classified as {}'.format(ans))
