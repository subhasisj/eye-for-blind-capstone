import streamlit as st
import tensorflow as tf
from PIL import Image


def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path



@st.cache
def load_model():
    pass

def main():
    st.title("Eye for the Blind - An Image Caption Generator")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")

def generate_captions():
    ...

if __name__ == "__main__":
    main()