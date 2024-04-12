import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.models import load_model


st.title('Vision Transformer Model Deployment')

model = load_model('C:/Users/KIIT01/Desktop/Minor Project/vit_faces95')
# inference_layer = TFSMLayer(saved_model_path, call_endpoint='serving_default')

def predict_image(image):
    image = image.resize((50, 50))
    image = np.array(image) / 255.0 # convert the image into numpy array & normalize in scale 0 to 1
    image = np.expand_dims(image, axis=0)  # expands the dimentions because the model expects to; set to 1 for a single image
    prediction = model.predict(image) # returns prediction probability
    return prediction


uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"]) # in byte stream


if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    
    prediction = predict_image(image)
    class_names = ['adhast', 'ajbake', 'apapou', 'apdavi', 'ardper', 'awjsud', 'boylee', 'bschap', 'cadugd', 'cdlarg', 'cfloro', 'cladam', 'cywan', 'dakram', 'damvo', 'darda', 'dfhodd', 'dgemen', 'gsmall', 'gstamo', 'gsvird', 'hcarpe', 'howar', 'hsgrim', 'ijfran', 'isbald', 'jbierl', 'jross', 'jserai', 'jshea', 'kbartl', 'kmbald', 'kouri', 'labenm', 'ldgodd', 'lidov', 'llambr', 'matth', 'mdchud', 'mizli', 'mkosto', 'mrhami', 'namart', 'ogefen', 'padnor', 'pajaco', 'papad', 'pcfry', 'pears', 'pjrand', 'rafox', 'rhaitk', 'rhnorm', 'riphil', 'rjdunc', 'sapere', 'sdwall', 'sgjday', 'sherbe', 'simm', 'sirmcb', 'sjcutt', 'sjkill', 'sjpalm', 'smille', 'theoc', 'theod', 'thgear', 'tjdyke', 'vanta', 'virvi', 'wjhugh']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Accuracy: {confidence*100}%")