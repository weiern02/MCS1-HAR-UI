import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import streamlit as st
import pandas as pd
import io
import re
import os
from math import sqrt, atan2
import matplotlib.pyplot as plt
import collections
from tensorflow.keras.models import load_model
import utils as utl
from preprocessing import process,csv_to_ampltude_plot,get_wavelet_cnn_model
from tensorflow.keras.models import load_model
from test_csv import check_shape,check_label_mac

activity_classes = ['standing', 'waving', 'clap', 'nopeople', 'squatting', 'jump', 'rubhand', 'punching', 'pushpull', 'twist']

def main():
    st.set_page_config(page_title='Human Activity Recognition App')
    utl.inject_custom_css()
    utl.navbar_component()

    page_selection = utl.get_current_route()
    
    if not page_selection:
        page_selection = "Prediction"

    if page_selection == "Prediction":
        option = st.selectbox(
            'Select an option',
            ('Front Only', 'Side Only', 'Both')
        )
        
        if option == 'Front Only' or option == "Side Only":

            model = get_wavelet_cnn_model()
            model.summary()
            model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
            if option == 'Front Only':
                model.load_weights("all_front_lpf_dwt_256.h5")
            else:
                model.load_weights("all_side_lpf_dwt_256.h5")

            if option == 'Front Only':
                st.markdown("<h4 style='text-align: center;'>Front Data</h4>", unsafe_allow_html=True)
                instruction = "Upload Front Data (.csv file)"
            else:
                st.markdown("<h4 style='text-align: center;'>Side Data</h4>", unsafe_allow_html=True)
                instruction = "Upload Side Data (.csv file)"
            
            uploaded_file = st.file_uploader(instruction, type=[".csv"])
            if uploaded_file is not None:
                file_path = uploaded_file.name 
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                if check_shape(file_path):
                    process(file_path)
                    img_path = 'action.jpg'
                    img = load_img(img_path, target_size=(256, 256))  
                    img_array = img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)  

                    predictions = model.predict(img_array)
                    predicted_class_index = np.argmax(predictions)
                    predicted_class = activity_classes[predicted_class_index]
                    expected = alpha_only_string = re.sub(r'[^a-zA-Z]', '', file_path[:-4])

                    st.write(f"<h5>Predicted Activity: <b>{predicted_class}</b> || Expected Activity: <b>{expected}</b></h5>", unsafe_allow_html=True)
                    if predicted_class!=expected:
                        st.write(f"<h4><font color='red'><b>Wrong Prediction</b></font></h4>", unsafe_allow_html=True)
                    else:
                        st.write(f"<h4><font color='green'><b>Correct Prediction</b></font></h4>", unsafe_allow_html=True)

                    table_data = []
                    for i, prob in enumerate(predictions[0]):
                        table_data.append([activity_classes[i], f"{prob:.4f}"])

                    df = pd.DataFrame(table_data)
                    df_transposed = df.T
                    new_header = df_transposed.iloc[0]
                    df_transposed = df_transposed[1:]
                    df_transposed.columns = new_header
                    dis_table = st.table(df_transposed)
                    csv_to_ampltude_plot(file_path)

                    st.write("Heatmap Plot")
                    st.image(img_path)
                    st.write("Amplitude Plot")
                    st.image("amplitude.jpg")

                    os.remove(img_path)
                    os.remove("amplitude.jpg")
                    os.remove(file_path)
                else:
                    st.write("<span style='color:red'>Invalid CSV!</span>", unsafe_allow_html=True)
                    os.remove(file_path)
        else:
            
            model = get_wavelet_cnn_model()
            model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
            model.load_weights("all_combine_lpf_dwt_256.h5")

            st.markdown("<h4 style='text-align: center;'>Front Data</h4>", unsafe_allow_html=True)
            uploaded_file_front = st.file_uploader("Upload Front Data (.csv file)", type=[".csv"])
            st.markdown("<h4 style='text-align: center;'>Side Data</h4>", unsafe_allow_html=True)
            uploaded_file_side = st.file_uploader("Upload Side Data (.csv file)", type=[".csv"])

            if uploaded_file_front is not None and uploaded_file_side is not None:

                file_path_front_name = "f_" + uploaded_file_front.name
                file_path_side_name = "s_" + uploaded_file_side.name
                
                with open(file_path_front_name, "wb") as f:
                    f.write(uploaded_file_front.getvalue())
                
                with open(file_path_side_name, "wb") as f:
                    f.write(uploaded_file_side.getvalue())

                if check_label_mac(file_path_front_name,file_path_side_name,re.sub(r'[^a-zA-Z]', '', file_path_front_name[:-4])[1:],re.sub(r'[^a-zA-Z]', '', file_path_side_name[:-4])[1:]):
                    process(file_path_front_name,file_path_side_name)
                    img_path = 'action.jpg'
                    img = load_img(img_path, target_size=(256, 256))  
                    img_array = img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) 
                    predictions = model.predict(img_array)
                    predicted_class_index = np.argmax(predictions)
                    predicted_class = activity_classes[predicted_class_index]
                    expected = re.sub(r'[^a-zA-Z]', '', file_path_front_name[:-4])[1:]

                    st.write(f"<h5>Predicted Activity: <b>{predicted_class}</b> || Expected Activity: <b>{expected}</b></h5>", unsafe_allow_html=True)
                    
                    if predicted_class!=expected:
                        st.write(f"<h4><font color='red'><b>Wrong Prediction</b></font></h4>", unsafe_allow_html=True)
                    else:
                        st.write(f"<h4><font color='green'><b>Correct Prediction</b></font></h4>", unsafe_allow_html=True)

                    table_data = []
                    for i, prob in enumerate(predictions[0]):
                        table_data.append([activity_classes[i], f"{prob:.4f}"])

                    df = pd.DataFrame(table_data)
                    df_transposed = df.T
                    new_header = df_transposed.iloc[0]
                    df_transposed = df_transposed[1:]
                    df_transposed.columns = new_header
                    dis_table = st.table(df_transposed)
                    
                    st.write("Heatmap Plot")
                    st.image(img_path)
                    csv_to_ampltude_plot(file_path_front_name)
                    os.rename("amplitude.jpg", "front.jpg")
                    st.write("Front Amplitude")
                    st.image("front.jpg")
                    csv_to_ampltude_plot(file_path_side_name)
                    st.write("Side Amplitude")
                    st.image("amplitude.jpg")

                    os.remove("front.jpg")
                    os.remove("amplitude.jpg")
                    os.remove(img_path)
                    os.remove(file_path_front_name)
                    os.remove(file_path_side_name)
                else:
                    st.write("<span style='color:red'>Mac Address should be different or Label should be same!</span>", unsafe_allow_html=True)
                    os.remove(file_path_front_name)
                    os.remove(file_path_side_name)



    elif page_selection == "Project Overview":
        """#### Enviromental Setup"""
        st.markdown("""
        <div style="text-align: justify">
        The data collection was done in a laboratory environment at Monash University Malaysia. The laboratory has the dimensions of 8 metres length 5 metres width  and 3 metres height). Four tripods with the data collection devices (ESP32-S3) mounted are placed inside the laboratory. The distance between each pair of ESP32-S3 devices is 1.5 metres. Both sender devices are connected to the PowerBank for power supply purposes after configuring it. Both receiver devices are connected to the researcher’s laptop using USB to Micro cable of 3 metres long to receive the data. The environment was only occupied by a participant (activity performer) to minimise any distraction or interference during the data collection. Besides the ESP32-S3 device set up in the laboratory, no other devices which would interfere with the data collection process including smartphone, smartwatch, keys etc are allowed during the data collection process.
        </div>
        """, unsafe_allow_html=True)
        st.write("-----------------------")
        st.image("./assets/images/setup.jpg",use_column_width=True)
        st.write("-----------------------")
        """#### Human Activity"""

        # Display images side by side using Streamlit columns layout
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])  # First row with 5 columns
        col6, col7, col8, col9, col10 = st.columns([1, 1, 1, 1, 1])     # Second row with 5 columns
    
        # Display first image in the first column
        with col1:
            st.image("./assets/images/noppl.gif",use_column_width=True ,caption="No People")

        # Display second image in the second column
        with col2:
            st.image("./assets/images/standing.gif", use_column_width=True,caption="Standing")

        # Display second image in the second column
        with col3:
            st.image("./assets/images/twist.gif", use_column_width=True,caption="Twistting")

        # Display second image in the second column
        with col4:
            st.image("./assets/images/squat.gif", use_column_width=True,caption="Squatting")

        # Display second image in the second column
        with col5:
            st.image("./assets/images/Jump.gif", use_column_width=True,caption="Jumping")

        # Display second image in the second column
        with col6:
            st.image("./assets/images/pushpull.gif", use_column_width=True, caption="Push pull")
  
        # Display second image in the second column
        with col7:
            st.image("./assets/images/clap.gif", use_column_width=True, caption="Clapping")

        # Display second image in the second column
        with col8:
            st.image("./assets/images/wave.gif", use_column_width=True, caption="Waving")

        # Display second image in the second column
        with col9:
            st.image("./assets/images/punch.gif", use_column_width=True, caption="Punching")

        # Display second image in the second column
        with col10:
            st.image("./assets/images/rubhand.gif", use_column_width=True, caption="Rub Hand")

if __name__ == "__main__":
    main()