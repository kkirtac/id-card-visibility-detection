import io

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import numpy as np
import cv2
import pandas as pd
import plotly.express as px

import streamlit as st

# interact with FastAPI endpoint
backend = "http://fastapi:8000/predict_visibility"


def process(image, server_url: str):

    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )

    return r


# construct UI layout
st.title("ID card visibility classification")

st.write(
    """Performs visibility classification of the ID card.
         This streamlit example uses a FastAPI service as backend.
         Visit this URL at `localhost:8000/docs` for FastAPI documentation."""
)  # description and instructions

uploaded_image = st.file_uploader("upload image")  # image upload widget

if uploaded_image:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # display the image
    st.image(opencv_image, caption='Image', channels="BGR")

if st.button("Classify visibility"):

    col1, col2 = st.columns(2)

    if uploaded_image:
        response = process(uploaded_image, backend)
        st.json(response.json())
        df = pd.DataFrame({k:[v] for k,v in response.json().items()})
        st.table(df)
        fig = px.bar(x=df.loc[0], y=df.columns, orientation='h')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    else:
        # handle case with no image
        st.write("Insert an image!")
