import pytest
import pandas as pd
from preprocessing import ESP32
import re

def check_shape(file_name):
    df = pd.read_csv(file_name)
    required_columns = ['type', 'id', 'mac', 'rssi', 'rate', 'sig_mode', 'mcs', 'local_timestamp', 'ant', 'sig_len', 'rx_state', 'len', 'first_word', 'data']
    # Assert that all required columns are in the DataFrame
    if not all(col in df.columns for col in required_columns):
        return False
    # Create an instance of ESP32 and filter the data
    matrix = ESP32(file_name).filter_by_sig_mode(1).get_csi()
    
    # Assert that each CSI data length is within the valid range
    return all(col in df.columns for col in required_columns) and all(1 <= len(matrix.csi_data[i]) <= 384 for i in range(len(matrix.csi_data)))

def check_label_mac(front, side,front_label=None,side_label=None):
    if front_label!=None and side_label!= None:
        if front_label!=side_label:
            return False
    else:
        pattern = r'([^\/]+)\.csv$'
        front_label = re.search(pattern, front)
        side_label = re.search(pattern, front)
        
        if front_label.group(1)!=side_label.group(1):
            return False

    front_data = pd.read_csv(front)
    side_data = pd.read_csv(side)
    
    # Ensure the shape of the files is valid
    if check_shape(front)==False or check_shape(side)==False:
        return False
    
    # Get the MAC addresses from the first row of each file
    front_mac = front_data.iloc[0]["mac"]
    side_mac = side_data.iloc[0]["mac"]
    
    if all(front_mac == i for i in front_data["mac"]) and all(side_mac == i for i in side_data["mac"]):
        return front_mac != side_mac
    else:
        return False

def test_check_shape():
    assert check_shape("./csv_test/front/clap2.csv")==True
    assert check_shape("./csv_test/side/clap2.csv")==True
    assert check_shape("./csv_test/front/clap16.csv")==True
    assert check_shape("./csv_test/side/pushpull24.csv")==True
    assert check_shape("./csv_test/CountriesData.csv")==False
    assert check_shape("./csv_test/incomplete.csv")==False

def test_check_label_mac():
    assert check_label_mac("./csv_test/front/clap2.csv","./csv_test/front/clap2.csv")==False
    assert check_label_mac("./csv_test/side/clap2.csv","./csv_test/front/clap2.csv")==True
    assert check_label_mac("./csv_test/side/clap2.csv","./csv_test/side/pushpull24.csv")==False

if __name__ == "__main__":
    pytest.main()
