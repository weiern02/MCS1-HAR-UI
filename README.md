# Setting Up Virtual Environment, Installing Required Packages & Run the Application

## Step 1: Create a Virtual Environment

Run the following command to create a virtual environment named `venv`:

```
python -m venv venv
```

## Step 2: Activate the Virtual Enviroment
```
venv\Scripts\activate
```

## If your python version is not 3.9.13, run the following command and IGNORE Step 3 and proceed to Step 4 after this:
Please go to your Microsoft Store to install Python 3.9 (this will not affect your original python version as this is runned inside the virtual environment)
```
py -3.9 -m pip install -r requirements.txt
py -3.9 -m pip install streamlit==1.33.0
py -3.9 -m pip install protobuf==3.20.0
py -3.9 -m streamlit run app.py
```

## Step 3: Install Required Packages
```
pip install -r requirements.txt
python install.py
```
Note: if facing pip version issue, try to run the following command and run Step 3 again: 

Note: if python install.py facing package dependency error, try to run python install.py again:
```
python.exe -m pip install --upgrade pip
```

## Step 4: Run the streamlit program
```
streamlit run app.py
```

## Step 5: App Access 
```
http://localhost:8501
```


