import streamlit as st
import base64
from streamlit.components.v1 import html

NAVBAR_PATHS = {
    'Model Prediction':'Prediction',
    'Project Overview': 'Project Overview'
}

def inject_custom_css():
    with open('assets/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def get_current_route():
    try:
        return st.query_params['nav']
    except:
        return None

def navbar_component():
    navbar_items = ''
    for key, value in NAVBAR_PATHS.items():
        navbar_items += (f'<a class="navitem" href="/?nav={value}" target="_self">{key}</a>')
    component = rf'''
            <nav class="container navbar" id="navbar">
                <ul class="navlist">
                {navbar_items}
            </nav>
            '''
    st.markdown(component, unsafe_allow_html=True)
