# Setting Up Virtual Environment, Installing Required Packages and Run the App
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
```

## Step 4: Run the streamlit program
```
streamlit run app.y
```

## Step 5: App Access
```
 http://localhost:8051
```

## Additional Step: Check the csv
```
pytest test_csv.py 
```