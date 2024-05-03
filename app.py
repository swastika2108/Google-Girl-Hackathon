import streamlit as st
from PIL import Image
import io
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from model_custom import model
import gmplot
import http.server
import socketserver

PORT = 5501

# visualize = False
src_coordinates_set = False
list_src_coordinates = []

# Create the map plotter:
apikey = ''
gmap = gmplot.GoogleMapPlotter(37.766956, -122.448481, 14, apikey=apikey)


st.set_page_config(page_title="Flood Detection Web App", page_icon="🌊")

SIZE = 128
unet = model.get_model()
unet.load_weights('unet.h5')


def generate_html(mask_difference):

    # Remove the noise
    thresh = cv2.erode(mask_difference, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    black_mask = thresh
    
    # Calculate Contours
    contours,_ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(test,contours,-1,(0,255,0),1)  
    # st.image(test)

    latitude_base = list_src_coordinates[0][0]
    longitude_base = list_src_coordinates[0][1]

    latitude_range = abs(list_src_coordinates[2][0] - list_src_coordinates[0][0])
    longitude_range = abs(list_src_coordinates[2][1] - list_src_coordinates[0][1])



    for contour in contours:
        print(contour.shape)
        
        temp_latitude = contour[:, 0, 0]
        temp_latitude = (( temp_latitude / SIZE ) * latitude_range ) + latitude_base
        temp_longitude = contour[:, 0, 1]
        temp_longitude = (( temp_longitude / SIZE ) * longitude_range ) + longitude_base

        gmap.polygon(temp_latitude,temp_longitude, color='cornflowerblue', edge_width=10)

    # Draw the map to an HTML file:
    gmap.draw('index.html')
    

    
def parse_string_to_list_of_tuples(input_string):
    # Split the string into a list where each element is a line
    lines = input_string.split('\n')
    
    # Split each line by space, convert each value to float, and convert to a tuple
    list_of_tuples = [tuple(float(value) for value in line.split()) for line in lines if line]
    return list_of_tuples

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

def predict_flood(mask_difference):
    
    # Remove the noise
    thresh = cv2.erode(mask_difference, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    black_mask = thresh
    

    # Create a red mask that highlights region with floods    
    red_mask = np.zeros ((SIZE, SIZE, 3), dtype=np.uint8)
    red_mask[thresh == 255] = [255,0,0]
    st.image(red_mask,caption="Predicted_Flood_Regions")
    
    # Calculate and show contours
    contours,_ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    test = np.zeros ((SIZE, SIZE, 3), dtype=np.uint8)
    cv2.drawContours(test,contours,-1,(0,255,0),1)  
    st.image(test)
    
    # Count the number of 1s in both masks
    count_before = np.count_nonzero(mask_before_thresholded == 255)
    count_after = np.count_nonzero(mask_after_thresholded == 255)
    
    # Calculate the percentage difference
    percentage_difference = abs(count_after - count_before) / (SIZE*SIZE) * 100

    # Define a threshold for what you consider a significant change
    SIGNIFICANT_THRESHOLD = 3  # Adjust this threshold as needed
    
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
        
input_string=st.text_area(label="Enter Coordinates of Corners")
if input_string:
    src_coordinates_set = True
    list_src_coordinates = parse_string_to_list_of_tuples(input_string)

if st.button("Generate Prediction",type="primary"):
    print("%s src_coordinates_set",src_coordinates_set)
    if image_uploaded and image_uploaded2 and src_coordinates_set:
        predicted_mask_before=process_and_predict(image_uploaded,unet)
        predicted_mask_after=process_and_predict(image_uploaded2,unet)
        col1,col2=st.columns(2)
        with col1:
            if predicted_mask_before is not None:
                st.image(predicted_mask_before,caption="Predicted_Mask_Before",use_column_width=True)
        with col2:
            if predicted_mask_after is not None:
                st.image(predicted_mask_after,caption="Predicted_Mask_After",use_column_width=True)

        # Compute the absolute difference between the two masks
        THRESHOLD = 0.3
        mask_before_thresholded  = np.where(predicted_mask_before > THRESHOLD, 255, 0)
        mask_after_thresholded  = np.where(predicted_mask_after > THRESHOLD, 255, 0)
        mask_difference = np.abs(mask_before_thresholded - mask_after_thresholded)
        mask_difference=np.uint8(mask_difference)


        predict_flood(mask_difference)
        generate_html(mask_difference)
        visualize = True
    elif not src_coordinates_set:
        st.warning("Please Enter Corner Coordinates")
    else:
        st.warning("Please upload both 'Before' and 'After' images to generate the analysis.")


url = "http://localhost:5501/index.html"

Handler = http.server.SimpleHTTPRequestHandler

if st.button('View Visualization',type='primary'):
    # if not visualize:
    #     st.warning("Please Run Generate Predictions before")
    # else:
    st.components.v1.iframe(url,width=800,height=500)

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print("serving at port", PORT)
            httpd.serve_forever()
except Exception as e:
    print("Server Already Running")