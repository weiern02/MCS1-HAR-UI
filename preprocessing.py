import pandas as pd
import numpy as np
import csv
from scipy.signal import butter, lfilter, freqz
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
from tensorflow.keras.models import load_model, model_from_json
from keras.models import model_from_json


import numpy as np
from matplotlib import pyplot as plt

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import add, concatenate


class ESP32:
    """Parse ESP32 Wi-Fi Channel State Information (CSI) obtained using ESP32 CSI Toolkit by Hernandez and Bulut.
    ESP32 CSI Toolkit: https://stevenmhernandez.github.io/ESP32-CSI-Tool/
    """

    # View README.md for more information on null subcarriers
    NULL_SUBCARRIERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 64, 65, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 382, 383]

    def __init__(self, csi_file):
        self.csi_file = csi_file
        self.__read_file()
    
    def __read_file(self):
        """Read RAW CSI file (.csv) using Pandas and return a Pandas dataframe
        """
        self.csi_df = pd.read_csv(self.csi_file)

    def seek_file(self):
        """Seek RAW CSI file
        """
        return self.csi_df

    def filter_by_sig_mode(self, sig_mode):
        """Filter CSI data by signal mode
        Args:  
            sig_mode (int):
            0 : Non - High Throughput Signals (non-HT)
            1 : HIgh Throughput Signals (HT)
        """
        self.csi_df = self.csi_df.loc[self.csi_df['sig_mode'] == sig_mode]
        return self

    def get_csi(self):
        """Read CSI string as Numpy array

        The CSI data collected by ESP32 contains channel frequency responses (CFR) represented by two signed bytes (imaginary, real) for each sub-carriers index
        The length (bytes) of the CSI sequency depends on the CFR type
        CFR consist of legacy long training field (LLTF), high-throughput LTF (HT-LTF), and space- time block code HT-LTF (STBC-HT-LTF)
        Ref: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi.html#wi-fi-channel-state-information

        NOTE: Not all 3 field may not be present (as represented in table and configuration)
        """
        raw_csi_data = self.csi_df['data'].copy()
        csi_data = np.array([np.fromstring(csi_datum.strip('[ ]'), dtype=int, sep = ',') for csi_datum in raw_csi_data])
        self.csi_data = csi_data
        return self

    # NOTE: Currently does not provide support for all signal subcarrier types
    def remove_null_subcarriers(self):
        """Remove NULL subcarriers from CSI
        """

        # Non-HT Signals (20 Mhz) - non STBC
        if self.csi_data.shape[1] == 128:
            remove_null_subcarriers = self.NULL_SUBCARRIERS[:24]
        # HT Signals (40 Mhz) - non STBC
        elif self.csi_data.shape[1] == 384:
            remove_null_subcarriers = self.NULL_SUBCARRIERS
        else:
            return self

        csi_data_T = self.csi_data.T
        csi_data_T_clean = np.delete(csi_data_T, remove_null_subcarriers, 0)
        csi_data_clean = csi_data_T_clean.T
        self.csi_data = csi_data_clean

        return self

    def get_amplitude_from_csi(self):
        """Calculate the Amplitude (or Magnitude) from CSI
        Ref: https://farside.ph.utexas.edu/teaching/315/Waveshtml/node88.html
        """
        amplitude = np.array([np.sqrt(data[::2]**2 + data[1::2]**2) for data in self.csi_data])
        self.amplitude = amplitude
        return self

    def get_phase_from_csi(self):
        """Calculate the Amplitude (or Magnitude) from CSI
        Ref: https://farside.ph.utexas.edu/teaching/315/Waveshtml/node88.html
        """
        phase = np.array([np.arctan2(data[::2], data[1::2]) for data in self.csi_data])
        self.phase = phase
        return self
     
def csv_simplify(file,output_filename):
    matrix = (ESP32(file)
                .filter_by_sig_mode(1)
                .get_csi()
                .remove_null_subcarriers()
                .get_amplitude_from_csi().amplitude)
    matrix = np.round(matrix, decimals = 5)
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in matrix:
            writer.writerow(row)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def lps_directory_to_heatmap(file, dim = 256, order = 4, fs = 40, cutoff = 3.5,):

    matrix = pd.read_csv(file, header=None) 
    matrix = matrix.astype(float)
                
    for column in matrix.columns:
        matrix[column] = butter_lowpass_filter(matrix[column], cutoff, fs, order=order)
    
    # Normalize data
    scaler = MinMaxScaler()
    scaled_matrix = scaler.fit_transform(matrix)
    
    # Resize data to 64x64 using OpenCV
    resized_matrix = cv2.resize(scaled_matrix, (dim, dim))

    # Reshape data for image
    image_data = resized_matrix.reshape(1, *resized_matrix.shape, 1)
    
    # Create ImageDataGenerator
    datagen = ImageDataGenerator(featurewise_center=False)
    
    # Fit generator to data
    datagen.fit(image_data)
    
    # Generate heatmap image
    for X_batch in datagen.flow(image_data, batch_size=1):
        heatmap_image = X_batch.squeeze()
        break  # only generate one batch
    
    # Plot heatmap
    plt.imshow(heatmap_image, cmap='jet', interpolation='nearest')
    plt.axis('off')
    plt.savefig("action.jpg", bbox_inches='tight')  # Save as .jpg file
    plt.close()

def lps_combine_directory_to_heatmap(front_file,side_file, dim=256, order=4, fs=40, cutoff=3.5):

    front = pd.read_csv(front_file, header=None).astype(float)
    side = pd.read_csv(side_file, header=None).astype(float)
    
    # Ensure both front and side data have the same length
    length = min(len(front), len(side))
    front = front.iloc[:length]
    side = side.iloc[:length]
                
    # Apply low-pass filter to each column
    for column in front.columns:
        front[column] = butter_lowpass_filter(front[column], cutoff, fs, order=order)
                    
    # Apply low-pass filter to each column
    for column in side.columns:
        side[column] = butter_lowpass_filter(side[column], cutoff, fs, order=order)               
                
    # Combine front and side data into a single matrix
    matrix = pd.concat([front, side], axis=1)
    
    # Normalize data
    scaler = MinMaxScaler()
    scaled_matrix = scaler.fit_transform(matrix)
    
    # Resize data to desired dimensions
    resized_matrix = cv2.resize(scaled_matrix, (dim, dim))

    # Reshape data for image
    image_data = resized_matrix.reshape(1, *resized_matrix.shape, 1)
    
    # Create ImageDataGenerator
    datagen = ImageDataGenerator(featurewise_center=False)
    
    # Fit generator to data
    datagen.fit(image_data)
    
    # Generate heatmap image
    for X_batch in datagen.flow(image_data, batch_size=1):
        heatmap_image = X_batch.squeeze()
        break  # only generate one batch
    
    # Plot heatmap
    plt.imshow(heatmap_image, cmap='jet', interpolation='nearest')
    plt.axis('off')
    plt.savefig("action.jpg", bbox_inches='tight')  # Save as .jpg file
    plt.close()

def process(file_name,file_name_2= None, both = False):
    if not both:
        csv_simplify(file_name,"simple.csv")
        lps_directory_to_heatmap("simple.csv")
        os.remove("simple.csv")
    else:
        csv_simplify(file_name,"simple1.csv")
        csv_simplify(file_name_2,"simple2.csv")
        lps_combine_directory_to_heatmap("simple1.csv","simple2.csv")
        os.remove(".\simple1.csv")
        os.remove(".\simple2.csv")

def amplitude_plot(amp, start_stamp = 0, num_packets = 1000, plot_name = ""):
    plt.clf()
    num_lines = amp.shape[1]
    print(num_lines)
    # initialize color map
    cmap = plt.cm.hsv
    # Create an array of equally spaced values between 0 and 1
    norm = np.linspace(0, 1, num_lines)
    # Create a ScalarMappable object to map values to colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))

    plt.figure(figsize=(15,10))
    df = np.asarray(amp, dtype=np.int32)

    # Can be changed to df[x] to plot sub-carrier x only (set color='r' also)
    for i in range(num_lines):   
        plt.plot(range(num_packets), df[start_stamp:start_stamp+num_packets, i], color= sm.to_rgba(norm[i]), linewidth = 0.5)
    #plt.plot(range(num_packets - len(amp), num_packets), df[:, subcarrier], color='b', linewidth = 0.5)

    #plt.colorbar(sm, label='Normalized Value')
    plt.xlabel("Packet")
    plt.ylabel("Amplitude")
    plt.xlim(0, num_packets)
    plt.title(f"Amplitude-Packet plot ({plot_name})")
    plt.savefig("amplitude.jpg", bbox_inches='tight')  # Save as .jpg file

def csv_to_ampltude_plot(file):
    # Filter requirements.
    order = 4
    fs = 40.0       # sample rate, Hz
    cutoff = 4  # desired cutoff frequency of the filter, Hz
    csv_simplify(file,"simple.csv")
    f = pd.read_csv("simple.csv")
    for column in f.columns:
        f[column] = butter_lowpass_filter(f[column], cutoff, fs, order=order)
    amplitude_plot(f, 0, 100, f"{file}")
    os.remove("simple.csv")

#csv_to_ampltude_plot("jump905.csv")


# batch operation usng tensor slice
def WaveletTransformAxisY(batch_img):
    odd_img  = batch_img[:,0::2]
    even_img = batch_img[:,1::2]
    L = (odd_img + even_img) / 2.0
    H = K.abs(odd_img - even_img)
    return L, H

def WaveletTransformAxisX(batch_img):
    # transpose + fliplr
    tmp_batch = K.permute_dimensions(batch_img, [0, 2, 1])[:,:,::-1]
    _dst_L, _dst_H = WaveletTransformAxisY(tmp_batch)
    # transpose + flipud
    dst_L = K.permute_dimensions(_dst_L, [0, 2, 1])[:,::-1,...]
    dst_H = K.permute_dimensions(_dst_H, [0, 2, 1])[:,::-1,...]
    return dst_L, dst_H
def Wavelet(batch_image):
    # make channel first image
    batch_image = K.permute_dimensions(batch_image, [0, 3, 1, 2])
    r = batch_image[:,0]
    g = batch_image[:,1]
    b = batch_image[:,2]

    # level 1 decomposition
    wavelet_L, wavelet_H = WaveletTransformAxisY(r)
    r_wavelet_LL, r_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    r_wavelet_HL, r_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_L, wavelet_H = WaveletTransformAxisY(g)
    g_wavelet_LL, g_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    g_wavelet_HL, g_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_L, wavelet_H = WaveletTransformAxisY(b)
    b_wavelet_LL, b_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    b_wavelet_HL, b_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_data = [r_wavelet_LL, r_wavelet_LH, r_wavelet_HL, r_wavelet_HH, 
                    g_wavelet_LL, g_wavelet_LH, g_wavelet_HL, g_wavelet_HH,
                    b_wavelet_LL, b_wavelet_LH, b_wavelet_HL, b_wavelet_HH]
    transform_batch = K.stack(wavelet_data, axis=1)

    # level 2 decomposition
    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(r_wavelet_LL)
    r_wavelet_LL2, r_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    r_wavelet_HL2, r_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(g_wavelet_LL)
    g_wavelet_LL2, g_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    g_wavelet_HL2, g_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(b_wavelet_LL)
    b_wavelet_LL2, b_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    b_wavelet_HL2, b_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)


    wavelet_data_l2 = [r_wavelet_LL2, r_wavelet_LH2, r_wavelet_HL2, r_wavelet_HH2, 
                    g_wavelet_LL2, g_wavelet_LH2, g_wavelet_HL2, g_wavelet_HH2,
                    b_wavelet_LL2, b_wavelet_LH2, b_wavelet_HL2, b_wavelet_HH2]
    transform_batch_l2 = K.stack(wavelet_data_l2, axis=1)

    # level 3 decomposition
    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(r_wavelet_LL2)
    r_wavelet_LL3, r_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    r_wavelet_HL3, r_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(g_wavelet_LL2)
    g_wavelet_LL3, g_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    g_wavelet_HL3, g_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(b_wavelet_LL2)
    b_wavelet_LL3, b_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    b_wavelet_HL3, b_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_data_l3 = [r_wavelet_LL3, r_wavelet_LH3, r_wavelet_HL3, r_wavelet_HH3, 
                    g_wavelet_LL3, g_wavelet_LH3, g_wavelet_HL3, g_wavelet_HH3,
                    b_wavelet_LL3, b_wavelet_LH3, b_wavelet_HL3, b_wavelet_HH3]
    transform_batch_l3 = K.stack(wavelet_data_l3, axis=1)

    # level 4 decomposition
    wavelet_L4, wavelet_H4 = WaveletTransformAxisY(r_wavelet_LL3)
    r_wavelet_LL4, r_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    r_wavelet_HL4, r_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

    wavelet_L4, wavelet_H4 = WaveletTransformAxisY(g_wavelet_LL3)
    g_wavelet_LL4, g_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    g_wavelet_HL4, g_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(b_wavelet_LL3)
    b_wavelet_LL4, b_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    b_wavelet_HL4, b_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)


    wavelet_data_l4 = [r_wavelet_LL4, r_wavelet_LH4, r_wavelet_HL4, r_wavelet_HH4, 
                    g_wavelet_LL4, g_wavelet_LH4, g_wavelet_HL4, g_wavelet_HH4,
                    b_wavelet_LL4, b_wavelet_LH4, b_wavelet_HL4, b_wavelet_HH4]
    transform_batch_l4 = K.stack(wavelet_data_l4, axis=1)

    # print('shape before')
    # print(transform_batch.shape)
    # print(transform_batch_l2.shape)
    # print(transform_batch_l3.shape)
    # print(transform_batch_l4.shape)

    decom_level_1 = K.permute_dimensions(transform_batch, [0, 2, 3, 1])
    decom_level_2 = K.permute_dimensions(transform_batch_l2, [0, 2, 3, 1])
    decom_level_3 = K.permute_dimensions(transform_batch_l3, [0, 2, 3, 1])
    decom_level_4 = K.permute_dimensions(transform_batch_l4, [0, 2, 3, 1])
    
    # print('shape after')
    # print(decom_level_1.shape)
    # print(decom_level_2.shape)
    # print(decom_level_3.shape)
    # print(decom_level_4.shape)
    return [decom_level_1, decom_level_2, decom_level_3, decom_level_4]



def Wavelet_out_shape(input_shapes):
    # print('in to shape')
    return [tuple([None, 112, 112, 12]), tuple([None, 56, 56, 12]), 
            tuple([None, 28, 28, 12]), tuple([None, 14, 14, 12])]

def get_wavelet_cnn_model(num_activities = 10):

    input_shape = 256, 256, 3

    input_ = Input(input_shape, name='the_input')
    # wavelet = Lambda(Wavelet, name='wavelet')
    wavelet = Lambda(Wavelet, Wavelet_out_shape, name='wavelet')
    input_l1, input_l2, input_l3, input_l4 = wavelet(input_)
    # print(input_l1)
    # print(input_l2)
    # print(input_l3)
    # print(input_l4)
    # level one decomposition starts
    conv_1 = Conv2D(64, kernel_size=(3, 3), padding='same', name='conv_1')(input_l1)
    norm_1 = BatchNormalization(name='norm_1')(conv_1)
    relu_1 = Activation('relu', name='relu_1')(norm_1)

    conv_1_2 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_1_2')(relu_1)
    norm_1_2 = BatchNormalization(name='norm_1_2')(conv_1_2)
    relu_1_2 = Activation('relu', name='relu_1_2')(norm_1_2)

    # level two decomposition starts
    conv_a = Conv2D(filters=64, kernel_size=(3, 3), padding='same', name='conv_a')(input_l2)
    norm_a = BatchNormalization(name='norm_a')(conv_a)
    relu_a = Activation('relu', name='relu_a')(norm_a)

    # concate level one and level two decomposition
    concate_level_2 = concatenate([relu_1_2, relu_a])
    conv_2 = Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_2')(concate_level_2)
    norm_2 = BatchNormalization(name='norm_2')(conv_2)
    relu_2 = Activation('relu', name='relu_2')(norm_2)

    conv_2_2 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_2_2')(relu_2)
    norm_2_2 = BatchNormalization(name='norm_2_2')(conv_2_2)
    relu_2_2 = Activation('relu', name='relu_2_2')(norm_2_2)

    # level three decomposition starts 
    conv_b = Conv2D(filters=64, kernel_size=(3, 3), padding='same', name='conv_b')(input_l3)
    norm_b = BatchNormalization(name='norm_b')(conv_b)
    relu_b = Activation('relu', name='relu_b')(norm_b)

    conv_b_2 = Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_b_2')(relu_b)
    norm_b_2 = BatchNormalization(name='norm_b_2')(conv_b_2)
    relu_b_2 = Activation('relu', name='relu_b_2')(norm_b_2)

    # concate level two and level three decomposition 
    concate_level_3 = concatenate([relu_2_2, relu_b_2])
    conv_3 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_3')(concate_level_3)
    norm_3 = BatchNormalization(name='nomr_3')(conv_3)
    relu_3 = Activation('relu', name='relu_3')(norm_3)

    conv_3_2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_3_2')(relu_3)
    norm_3_2 = BatchNormalization(name='norm_3_2')(conv_3_2)
    relu_3_2 = Activation('relu', name='relu_3_2')(norm_3_2)

    # level four decomposition start
    conv_c = Conv2D(64, kernel_size=(3, 3), padding='same', name='conv_c')(input_l4)
    norm_c = BatchNormalization(name='norm_c')(conv_c)
    relu_c = Activation('relu', name='relu_c')(norm_c)

    conv_c_2 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_c_2')(relu_c)
    norm_c_2 = BatchNormalization(name='norm_c_2')(conv_c_2)
    relu_c_2 = Activation('relu', name='relu_c_2')(norm_c_2)

    conv_c_3 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_c_3')(relu_c_2)
    norm_c_3 = BatchNormalization(name='norm_c_3')(conv_c_3)
    relu_c_3 = Activation('relu', name='relu_c_3')(norm_c_3)

    # concate level level three and level four decomposition
    concate_level_4 = concatenate([relu_3_2, relu_c_3])
    conv_4 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_4')(concate_level_4)
    norm_4 = BatchNormalization(name='norm_4')(conv_4)
    relu_4 = Activation('relu', name='relu_4')(norm_4)

    conv_4_2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_4_2')(relu_4)
    norm_4_2 = BatchNormalization(name='norm_4_2')(conv_4_2)
    relu_4_2 = Activation('relu', name='relu_4_2')(norm_4_2)

    conv_5_1 = Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_5_1')(relu_4_2)
    norm_5_1 = BatchNormalization(name='norm_5_1')(conv_5_1)
    relu_5_1 = Activation('relu', name='relu_5_1')(norm_5_1)

    pool_5_1 = AveragePooling2D(pool_size=(7, 7), strides=1, padding='same', name='avg_pool_5_1')(relu_5_1)
    flat_5_1 = Flatten(name='flat_5_1')(pool_5_1) 
    # Add a Reshape layer to ensure the correct input shape for the Dense layer
    #reshape_5_1 = Reshape((6272,), name='reshape_5_1')(flat_5_1)

    #fc_5 = Dense(2048, name='fc_5')(reshape_5_1)
    fc_5 = Dense(2048, name='fc_5')(flat_5_1)
    norm_5 = BatchNormalization(name='norm_5')(fc_5)
    relu_5 = Activation('relu', name='relu_5')(norm_5)
    drop_5 = Dropout(0.5, name='drop_5')(relu_5)

    fc_6 = Dense(2048, name='fc_6')(drop_5)
    norm_6 = BatchNormalization(name='norm_6')(fc_6)
    relu_6 = Activation('relu', name='relu_6')(norm_6)
    drop_6 = Dropout(0.5, name='drop_6')(relu_6)

    output = Dense(num_activities, activation='softmax', name='fc_7')(drop_6)

    model = Model(inputs=input_, outputs=output)
    #model.summary()
    #plot_model(model, to_file='wavelet_cnn_0.5.png')

    return model