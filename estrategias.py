import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import re
import datetime
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from app_funciones import *


##########################################################
##########################################################
##########################################################

def estrategias_page():
    st.header("Página: Estrategias")
    st.write("Aquí puedes documentar y explorar estrategias.")
    estrategia = st.text_area("Describe una estrategia:")
    if estrategia:
        st.write("Tu estrategia:")
        st.write(estrategia) 
    

    with open("/workspaces/Proyecto_dcd/estrategias_final.md", "r") as file:
        contenido_markdown_estrategias = file.read()
    st.markdown(contenido_markdown_estrategias)