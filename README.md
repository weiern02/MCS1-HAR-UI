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
