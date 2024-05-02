import streamlit as st
from PIL import Image
import io
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from model_custom import model


SIZE = 128
unet = model.get_model()
unet.load_weights('UNet-01.h5')

# Set page configuration
st.set_page_config(page_title="Flood Detection Web App", page_icon="🌊")
# st.set_option('deprecation.showPyplotGlobalUse', False)

# Define the image processing function
def process_image(uploaded_file):
    try:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        return None

#Define the function to process and predict the image
def process_and_predict(image, model, size=128):
    if image is not None:
        image_array = img_to_array(image).astype('float32') / 255.
        image_resized = cv2.resize(image_array, (size, size), interpolation=cv2.INTER_AREA)
        image_expanded = np.expand_dims(image_resized, axis=0)
        pred_mask = unet.predict(image_expanded).reshape(size, size)
        return pred_mask
    return None

def predict_flood(mask_before, mask_after):
    
    # Compute the absolute difference between the two masks
    THRESHOLD = 0.3
    mask_before_thresholded  = np.where(mask_before > THRESHOLD, 255, 0)
    mask_after_thresholded  = np.where(mask_after > THRESHOLD, 255, 0)
    mask_difference = np.abs(mask_before_thresholded - mask_after_thresholded)
    mask_difference=np.uint8(mask_difference)

    # Remove the noise
    thresh = cv2.erode(mask_difference, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    # Create a red mask that highlights region with floods    
    rgb_mask = np.zeros ((SIZE, SIZE, 3), dtype=np.uint8)
    rgb_mask[thresh == 255] = [255,0,0]
    st.image(rgb_mask)
    
    
    # Count the number of 1s in both masks
    count_before = np.count_nonzero(mask_before_thresholded == 255)
    count_after = np.count_nonzero(mask_after_thresholded == 255)
    
    # Calculate the percentage difference
    percentage_difference = abs(count_after - count_before) / (SIZE*SIZE) * 100

    # Define a threshold for what you consider a significant change
    SIGNIFICANT_THRESHOLD = 10  # Adjust this threshold as needed
    
    # Display the results
    st.text(f"Percentage difference: {percentage_difference:.2f}%")
    
    # Check if the percentage difference is significant
    if percentage_difference > SIGNIFICANT_THRESHOLD:
        st.text("Flood is detected")
    else:
        st.text("Flood Not Detected")


# Header for the app
st.header("Semantic Segmentation of Flood Events using U-Net")
# File uploaders for 'Before' and 'After' images
st.subheader("Upload Images for Analysis")
col1, col2 = st.columns(2)

with col1:
    st.text("Upload 'Before' Image")
    uploaded_file1 = st.file_uploader("", type=["jpeg", "png", "jpg"],key="before")
    image_uploaded=process_image(uploaded_file1)
    if image_uploaded:
        st.image(image_uploaded)
 
with col2:
    st.text("Upload 'After' Image")
    uploaded_file2 = st.file_uploader("", type=["jpeg", "png", "jpg"], key="after")
    image_uploaded2=process_image(uploaded_file2)
    if image_uploaded2 is not None:
        st.image(image_uploaded2)

if st.button("Generate Prediction",type="primary"):
    if image_uploaded and image_uploaded2:
        predicted_mask_before=process_and_predict(image_uploaded,unet)
        predicted_mask_after=process_and_predict(image_uploaded2,unet)
        col1,col2=st.columns(2)
        with col1:
            if predicted_mask_before is not None:
                st.image(predicted_mask_before,caption="Predicted_Mask_Before",use_column_width=True)
        with col2:
            if predicted_mask_after is not None:
                st.image(predicted_mask_after,caption="Predicted_Mask_After",use_column_width=True)
        predict_flood(predicted_mask_before,predicted_mask_after)
        


    else:
        st.warning("Please upload both 'Before' and 'After' images to generate the analysis.")
