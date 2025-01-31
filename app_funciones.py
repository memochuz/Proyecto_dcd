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
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.ticker as ticker  # Importar formateador de números

######################################################################
########################### FUNCIONES ################################
######################################################################

# Función de limpieza
def limpieza(df):
    df = df.dropna(subset=["CustomerNo"])
    df = df[~df['TransactionNo'].str.contains('^C')]
    df['ProductName'] = df['ProductName'].str.lower()
    patron = r"[?¿]"
    regex = re.compile(patron,re.IGNORECASE)
    mask = ~(df["ProductName"].str.contains(regex) & df["ProductName"].str.len() >= 4)
    df = df[mask]
    df["TotalSpent"] = df["Price"] * df["Quantity"]
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Creación de la tabla RFM aumentada 
def rfm_df(df):
    latest_date = df['Date'].max()
    rfm = df.groupby('CustomerNo').agg({
        'Date': lambda x: (latest_date - x.max()).days,  # Recency
        'TransactionNo': 'count',                        # Frequency
        'TotalSpent': 'sum'                              # Monetary
        }).rename(columns={
        'Date': 'Recency',
        'TransactionNo': 'Frequency',
        'TotalSpent': 'Monetary'
        }).reset_index()
    rfm['AvgPurchaseSize'] = rfm['Monetary'] / rfm['Frequency'] # Promedio $ en compras
    rfm['AvgTimeBetweenPurchases']= df.groupby('CustomerNo')['Date'].apply(
        lambda x: (x.max() - x.min()).days / (len(x) - 1) if len(x) > 1 else 0
        ).reset_index(drop=True)
    return rfm

# Estandarizar los datos
def std_data(df):
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return scaled_data

# Crear una gráfica de codo
def elbow_graph(df,*columns):
    df = df.loc[:, list(columns)]
    # Aplicar estilo limpio
    plt.style.use("dark_background")  
    # Crear figura para Streamlit
    fig_elb, ax_elb= plt.subplots(figsize=(8, 6))
    vis_elb = KElbowVisualizer(KMeans(n_init=20), k=(1, 11), timings=False, ax=ax_elb)
    vis_elb.fit(df)
    vis_elb.poof()  # Renderiza el gráfico
    # Cambiar el color de la línea vertical
    for line in ax_elb.lines:
        if line.get_linestyle() == '--': 
            line.set_color('#FF6961') 
            line.set_linewidth(4)
    # Personalizar etiquetas y títulos
    ax_elb.set_title("Elbow Method para Clustering", fontsize=16, color="black")
    ax_elb.set_xlabel("Número de Clusters", fontsize=12, color="gray")
    ax_elb.set_ylabel("Distorsión", fontsize=12, color="gray")
    
    return fig_elb, vis_elb.elbow_value_


# Kmeans
def kmeans(df,clusters,*columns):
    df = df.loc[:, list(columns)]
    km =KMeans(n_clusters=clusters,  random_state=33)
    # entrenamiento (fit)
    km.fit(df)
    labels = km.predict(df) 
    centroides = km.cluster_centers_
    return labels, km

#DBScan
def dbscan_clustering(df, eps=0.5, min_samples=5,*columns):
    df = df.loc[:, list(columns)]
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_model.fit(df)
    labels = dbscan_model.labels_
    return labels, dbscan_model

#Gaussian Mix
def gaussian_mixture(df, n_clusters=3,*columns):
    df = df.loc[:, list(columns)]
    gmm = GaussianMixture(n_components=n_clusters, random_state=33)
    gmm.fit(df)
    labels = gmm.predict(df)
    return labels, gmm

# métricas
def evaluate_clustering(df, labels,*columns):
    
    df = df.loc[:, list(columns)]
    if len(set(labels)) > 1:  # Asegurarse de que haya más de un clúster
        silhouette = silhouette_score(df, labels)
        calinski_harabasz = calinski_harabasz_score(df, labels)
    else:
        silhouette = "N/A"
        calinski_harabasz = "N/A"
    
    return {
        "Silhouette Score": silhouette,
        "Calinski-Harabasz Index": calinski_harabasz
    }

# PCA
def pca(df,n_componentes,*columns):
    df = df.loc[:, list(columns)]
    pca= PCA(n_components = n_componentes, random_state=33)
    return pca.fit_transform(df)

# 3d graph
def graph_3d(df,titulo,labels,labelx,labely,labelz,*columns):
    df = df.loc[:, list(columns)]
    plt.style.use("dark_background")  
    fig = plt.figure(figsize=(30, 7))
    ax  = fig.add_subplot(111, projection='3d')
    x   = df.iloc[:, 0]
    y   = df.iloc[:, 1]
    z   = df.iloc[:, 2]
    # Crear scatter plot
    img = ax.scatter(x, y, z, c=labels, cmap='viridis')
    ax.set_title(titulo, fontsize=16)
    # Crear una leyenda personalizada basada en los grupos
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())
    unique_groups = np.unique(labels)
    for group in unique_groups:
        ax.scatter([], [], [], color=cmap(norm(group)), label=f'Grupo {group}')
    # Añadir la leyenda al gráfico
    ax.legend(title="Clusters", loc='upper left', bbox_to_anchor=(1.05, 1))
    # Ajustar etiquetas de los ejes
    ax.set_xlabel(labelx, labelpad=10, fontsize=12)
    ax.set_ylabel(labely, labelpad=10, fontsize=12)
    ax.set_zlabel(labelz, labelpad=10, fontsize=12)
    # Mostrar el gráfico
    return fig

# Gráfica de barras para pca_kmeans
def bar_graph(df,titulo,leyenda,labelx,labely,*columns):
    df = df.loc[:, list(columns)]
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))  # Crear figura y ejes
    df.plot(kind='bar', ax=ax, edgecolor='black', linewidth=2)
    # Personalización del gráfico
    ax.set_title(titulo, fontsize=16)
    ax.set_xlabel(labelx, fontsize=14)
    ax.set_ylabel(labely, fontsize=14)
    ax.set_xticks(range(len(df.index)))  
    ax.set_xticklabels(df.index, rotation=0, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(title=leyenda, fontsize=12, title_fontsize=14, loc='upper right',bbox_to_anchor=(1.15, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Mostrar en Streamlit
    return fig

# grafica de barra para el análisis
def bars(df,titulo,labelx,labely):
    plt.style.use("dark_background")  
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df.iloc[:,0], df.iloc[:,1],edgecolor='black', linewidth=2)
    # Personalizar etiquetas y títulos
    ax.set_xlabel(labelx, fontsize=12)
    ax.set_ylabel(labely, fontsize=12)
    ax.set_title(titulo, fontsize=14)
    ax.tick_params(axis='x', rotation=90) 
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    return fig

def graph_3d_segements(df,labels,labelx,labely,labelz,cluster_labels,*columns,):
    df = df.loc[:, list(columns)]
    plt.style.use("dark_background")  
    fig = plt.figure(figsize=(30, 7))
    ax  = fig.add_subplot(111, projection='3d')
    x   = df.iloc[:, 0]
    y   = df.iloc[:, 1]
    z   = df.iloc[:, 2]
    # Crear scatter plot
    img = ax.scatter(x, y, z, c=labels, cmap='viridis')
    # Crear una leyenda personalizada basada en los grupos
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())
    unique_groups = np.unique(labels)
    for group in unique_groups:
        ax.scatter([], [], [], color=cmap(norm(group)), label=f'Grupo {cluster_labels[group]}')
    # Añadir la leyenda al gráfico
    ax.legend(title="Clusters", loc='upper left', bbox_to_anchor=(1.05, 1))
    # Ajustar etiquetas de los ejes
    ax.set_xlabel(labelx, labelpad=10, fontsize=12)
    ax.set_ylabel(labely, labelpad=10, fontsize=12)
    ax.set_zlabel(labelz, labelpad=10, fontsize=12)
    plt.tight_layout()
    # Mostrar el gráfico
    return fig