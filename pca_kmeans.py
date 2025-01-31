import streamlit as st
from app_funciones import *
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
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import DBSCAN


##########################################################
##########################################################
##########################################################

def pca_kmeans_page():
    st.header("Página: PCA y Kmeans")
    st.write("Aquí puedes entrenar y evaluar modelos de machine learning.")
    # st.subheader("Escojamos el algoritmo de clusterización que más nos convenga.")

    st.markdown("### Creamos la tablas RFM")
    st.write("Calcular la tabla RFM aumentada y estandarizada.")

    # Tabla RFM Y RFM estandarizada
    generar_rfm = st.checkbox("Generar RFM DataFrame")
    if generar_rfm  and st.session_state.datos is not None:
        st.session_state.rfm = rfm_df(st.session_state.datos)
        st.session_state.rfm_std = std_data(st.session_state.rfm.iloc[:,1:6])
        st.session_state.rfm_std.insert(0,"CustomerNo",st.session_state.rfm["CustomerNo"])
        st.write("RFM y RFM estandarizada:")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Tabla RFM")
            st.dataframe(st.session_state.rfm)
        with col2:
            st.subheader("Tabla RFM estandarizada")
            st.dataframe(st.session_state.rfm_std)

    ####################################################

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### Flujo para la selección del modelo")

    col1 , col2, col3 = st.columns([1,2,1])

    with col2:
        st.image("flujo.png")

    ####################################################
    ####################################################
    ####################################################
    ####################################################
    ####################################################

    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("Modelo 1")
    st.markdown("### Características")

    st.write("Escoja 3 características que desea usar de la tabla RFM estandarizada:")
    columns_select = st.multiselect("Selecciona las características de RFM estandarizada",
                                 options = ["Recency","Frequency","Monetary","AvgPurchaseSize","AvgTimeBetweenPurchases"],
                                 default = ["Recency","Frequency","Monetary"]
                                 )

    ### FUNCIONÓ
    if columns_select and st.session_state.rfm_std is not None:
        rfm_aux = st.session_state.rfm_std.loc[:, list(columns_select)]
    else:
        st.warning("Por favor, genere el RFM dataframe.")
        rfm_aux = None
    ####


    st.subheader("Número óptimo de clusters para Kmeans y Gaussian Mixture:")

    st.write("Usemos la gráfica de codo para escoger el número de clusters para Kmeans y Gaussian Mixture.")

    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        ### FUNCIONÓ
        if columns_select and st.session_state.rfm_std is not None:
            elbow_fig, clusters = elbow_graph(st.session_state.rfm_std,*columns_select)
            st.pyplot(elbow_fig)
        else:
            st.warning("Por favor, genere el RFM dataframe.")
        ###
        
        
    ####################################################
    st.subheader("Comparación de hiperparámetros para DBSCAN")
    st.write("Por último se escogen los hiperparámetros de dbscan en base a la siguiente tabla:")

    parametros_grid = pd.DataFrame({
        "eps":[0.1,0.2,0.3,0.5],
        "min_samples":[3,5,8,None]
        })
    
    param_grid = {
        "eps":[0.1,0.2,0.3,0.5],
        "min_samples":[3,5,8]
        }

    # AQUÍ DEBO PONER LAS ESTADISTICAS DE LA RMF ESTANDARIZADA PARA DECIR QUE POR ESO ESCOGI ESAS DISTANCIAS
    
    st.dataframe(parametros_grid)
    
    ### OJO
    ####################################################
    if isinstance(rfm_aux, pd.DataFrame):
        grid = ParameterGrid(param_grid)
        # Lista para almacenar resultados
        resultados_dbscan = []

        ###

        for params in grid:
            dbscan = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
            labels_dbscan = dbscan.fit_predict(rfm_aux)
            # Número de clusters válidos
            n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)  # Excluir ruido (-1)
            if n_clusters_dbscan > 1:  # Solo evaluamos si hay más de un cluster
                silhouette_avg = silhouette_score(rfm_aux, labels_dbscan)
                ch_score = calinski_harabasz_score(rfm_aux, labels_dbscan)
            else:
                silhouette_avg = -1  # Penalización si no hay clusters válidos
                ch_score = -1
            # Guardar resultados
            resultados_dbscan.append({
                "eps": params["eps"],
                "min_samples": params["min_samples"],
                "n_clusters": n_clusters_dbscan,
                "silhouette_score": silhouette_avg,
                "calinski_harabasz_score": ch_score
            })
            resultados_dbscan_df = pd.DataFrame(resultados_dbscan)
            mejor_dbscan_df = resultados_dbscan_df.sort_values(
                by=["silhouette_score", "calinski_harabasz_score"],  # Ordenamos por ambas métricas
                ascending=[False, False]  # Queremos valores más altos primero
                )

    
        ####################################################

        # Convertir los resultados en un DataFrame y ordenar por la métrica deseada
        # resultados_dbscan_df = pd.DataFrame(resultados_dbscan).sort_values(by="silhouette_score", ascending=False)
        # st.dataframe(resultados_dbscan_df.head())
        st.markdown(
            """
            Recordemos que:

            1. **silhouette_score:**

                - Mide qué tan separados están los clusters y qué tan compactos son.
                - Rango: [-1, 1] (valores cercanos a 1 son mejores).

            2. **calinski_harabasz_score:**

                - Evalúa la dispersión dentro de los clústeres y la dispersión entre los clústeres.
                - Valores más altos indican mejor separación.
            """
        )
        st.write("Los mejores parámetros en base a lo anterior son:")
        st.write(mejor_dbscan_df.head(1))

        ####################################################

        st.subheader("Comparación de los tres modelos")

        # Modelos y Scores

        y_hat_kmean, km = kmeans(st.session_state.rfm_std,clusters,*columns_select)
        score_kmean = evaluate_clustering(
                    st.session_state.rfm_std,
                    y_hat_kmean,
                    *columns_select
                    )

        y_hat_gmm, gmm = gaussian_mixture(st.session_state.rfm_std,clusters,*columns_select)
        score_gmm = evaluate_clustering(
                    st.session_state.rfm_std,
                    y_hat_gmm,
                    *columns_select
                    )

        y_hat_dbscan, dbscan_model = dbscan_clustering(st.session_state.rfm_std,
                                      mejor_dbscan_df.iloc[0,0],
                                      mejor_dbscan_df.iloc[0,1],
                                      *columns_select
                                      )
        score_dbscan = evaluate_clustering(
                    st.session_state.rfm_std,
                    y_hat_dbscan,
                    *columns_select
                    )

        ####################################################
        # No podemos graficar pero podemos obtener de todas maneras las clusterizaciones y los scores
        if len(columns_select) == 3:

            col1, col2, col3 = st.columns([1,1,1])


            with col1:
                st.markdown("### Kmeans")
                # y_hat_kmean, km = kmeans(st.session_state.rfm_std,clusters,*columns_select)

                agrupamiento_kmean = graph_3d(
                    st.session_state.rfm_std,
                    "Kmeans",
                    y_hat_kmean,
                    columns_select[0],
                    columns_select[1],
                    columns_select[2],
                    *columns_select
                    )
                st.pyplot(agrupamiento_kmean)

                st.write("Scores")

                # score_kmean = evaluate_clustering(
                #     st.session_state.rfm_std,
                #     y_hat_kmean,
                #     *columns_select
                #     )

                st.dataframe(pd.DataFrame.from_dict(score_kmean, orient="index", columns=["Valor"]))




            with col2:

                st.markdown("### Gaussian Mixture")

                # y_hat_gmm, gmm = gaussian_mixture(st.session_state.rfm_std,clusters,*columns_select)

                agrupamiento_gmm = graph_3d(
                    st.session_state.rfm_std,
                    "Gaussian Mixture",
                    y_hat_gmm,
                    columns_select[0],
                    columns_select[1],
                    columns_select[2],
                    *columns_select
                    )
                st.pyplot(agrupamiento_gmm)

                st.write("Scores")

                # score_gmm = evaluate_clustering(
                #     st.session_state.rfm_std,
                #     y_hat_gmm,
                #     *columns_select
                #     )

                st.dataframe(pd.DataFrame.from_dict(score_gmm, orient="index", columns=["Valor"]))
            with col3:

                st.markdown("###  DBSCAN")
                # y_hat_dbscan, dbscan_model = dbscan_clustering(st.session_state.rfm_std,
                #                       mejor_dbscan_df.iloc[0,0],
                #                       mejor_dbscan_df.iloc[0,1],
                #                       *columns_select
                #                       )

                agrupamiento_dbscan = graph_3d(
                    st.session_state.rfm_std,
                     "DBSCAN",
                    y_hat_dbscan,
                    columns_select[0],
                    columns_select[1],
                    columns_select[2],
                    *columns_select
                    )
                st.pyplot(agrupamiento_dbscan)

                st.write("Scores")

                # score_dbscan = evaluate_clustering(
                #     st.session_state.rfm_std,
                #     y_hat_dbscan,
                #     *columns_select
                #     )

                st.dataframe(pd.DataFrame.from_dict(score_dbscan, orient="index", columns=["Valor"]))
        else:

            col1, col2, col3 = st.columns([1,1,1])

            with col1:
                st.markdown("### Kmeans")
                st.dataframe(pd.DataFrame.from_dict(score_kmean, orient="index", columns=["Valor"]))

            with col2:
                st.markdown("### Gaussian Mixture")
                st.dataframe(pd.DataFrame.from_dict(score_gmm, orient="index", columns=["Valor"]))

            with col3:
                st.markdown("### DBSCAN")
                st.dataframe(pd.DataFrame.from_dict(score_dbscan, orient="index", columns=["Valor"]))



        st.markdown(
            """
            Si se tienen dos modelos con Silhouette Scores similares, pero el Calinski-Harabasz Score es claramente más alto en uno,  
            lo ideal es escoger el modelo con el mayor Calinski-Harabasz Score.
            """
        )

        ####################################################

        st.subheader("Elección del Modelo 1")

        st.markdown("Basados en los scores anteriores tenemos que el mejor modelo en este caso es:")

        modelos_1 = st.selectbox("Selecciona el modelo:" , ["Kmeans","Gaussian Mixture", "DBSCAN"])


        if modelos_1:
            if modelos_1 == "Kmeans":
                y_hat_modelo_1 = y_hat_kmean
                score_modelo_1 = score_kmean
                modelo_1 = km
                columns_select_modelo_1 = columns_select
            elif modelos_1 == "Gaussian Mixture":
                y_hat_modelo_1 = y_hat_gmm
                score_modelo_1 = score_gmm
                modelo_1 = gmm
                columns_select_modelo_1 = columns_select

            else:
                y_hat_modelo_1 = y_hat_dbscan
                score_modelo_1 = score_dbscan
                modelo_1 = dbscan_model
                columns_select_modelo_1 = columns_select

        ####################################################
        ####################################################
        ####################################################
        ####################################################
        ####################################################

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Modelo 2")
        st.write("Usamos la tabla RFM estandarizada con todas las características.")
        st.subheader("Aplicación de PCA")

        # PCA
        pca= PCA(n_components = 3, random_state=33)
        pca_columns = ["Componente Principal 1",
                       "Componente Principal 2",
                       "Componente Principal 3"
                       ]
        datos_pca = pd.DataFrame(pca.fit_transform(st.session_state.rfm_std.iloc[:,1:6]),
                                 columns = pca_columns)




        st.dataframe(datos_pca)

        ####################################################

        st.subheader("Número óptimo de clusters para Kmeans y Gaussian Mixture:")

        elbow_fig_2, clusters_2 = elbow_graph(datos_pca,*pca_columns)

        col1,col2,col3 = st.columns([1,2,1])

        with col2:
            st.pyplot(elbow_fig)

        ####################################################
        st.subheader("Comparación de hiperparámetros para DBSCAN")
        st.write("Por último se escogen los hiperparámetros de dbscan en base a la siguiente tabla:")

        parametros_grid_2 = pd.DataFrame({
            "eps":[0.1,0.2,0.3,0.5],
            "min_samples":[3,5,8,None]
            })

        param_grid_2 = {
            "eps":[0.1,0.2,0.3,0.5],
            "min_samples":[3,5,8]
            }

        # AQUÍ DEBO PONER LAS ESTADISTICAS DE LA RMF ESTANDARIZADA PARA DECIR QUE POR ESO ESCOGI ESAS DISTANCIAS

        st.dataframe(parametros_grid_2)

        ####################################################

        rfm_aux_2 = st.session_state.rfm_std.iloc[:,1:6]
        grid_2 = ParameterGrid(param_grid_2)

        # Lista para almacenar resultados
        resultados_dbscan_2 = []

        for params in grid_2:
            dbscan_2 = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
            labels_dbscan_2 = dbscan_2.fit_predict(rfm_aux_2)

            # Número de clusters válidos
            n_clusters_dbscan_2 = len(set(labels_dbscan_2)) - (1 if -1 in labels_dbscan_2 else 0)  # Excluir ruido (-1)

            if n_clusters_dbscan_2 > 1:  # Solo evaluamos si hay más de un cluster
                silhouette_avg_2 = silhouette_score(rfm_aux_2, labels_dbscan_2)
                ch_score_2 = calinski_harabasz_score(rfm_aux_2, labels_dbscan_2)
            else:
                silhouette_avg_2 = -1  # Penalización si no hay clusters válidos
                ch_score_2 = -1

            # Guardar resultados
            resultados_dbscan_2.append({
                "eps": params["eps"],
                "min_samples": params["min_samples"],
                "n_clusters": n_clusters_dbscan_2,
                "silhouette_score": silhouette_avg_2,
                "calinski_harabasz_score": ch_score_2
            })

            resultados_dbscan_df_2 = pd.DataFrame(resultados_dbscan_2)
            mejor_dbscan_df_2 = resultados_dbscan_df_2.sort_values(
                by=["silhouette_score", "calinski_harabasz_score"],  # Ordenamos por ambas métricas
                ascending=[False, False]  # Queremos valores más altos primero
                )
        ####################################################

        # Modelos y Scores

        y_hat_kmean_2, km_2 = kmeans(datos_pca,clusters_2,*pca_columns)
        score_kmean_2 = evaluate_clustering(
                    datos_pca,
                    y_hat_kmean_2,
                    *pca_columns
                    )

        y_hat_gmm_2, gmm_2 = gaussian_mixture(datos_pca,clusters_2,*pca_columns)
        score_gmm_2 = evaluate_clustering(
                    datos_pca,
                    y_hat_gmm_2,
                    *pca_columns
                    )

        y_hat_dbscan_2, dbscan_model_2 = dbscan_clustering(datos_pca,
                                      mejor_dbscan_df_2.iloc[0,0],
                                      mejor_dbscan_df_2.iloc[0,1],
                                      *pca_columns
                                      )
        score_dbscan_2 = evaluate_clustering(
                    datos_pca,
                    y_hat_dbscan_2,
                    *pca_columns
                    )

        ####################################################
        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            st.markdown("### Kmeans")

            agrupamiento_kmean_2 = graph_3d(
                    datos_pca,
                    "Kmeans",
                    y_hat_kmean_2,
                    columns_select[0],
                    columns_select[1],
                    columns_select[2],
                    *pca_columns
                    )
            st.pyplot(agrupamiento_kmean_2)

            st.dataframe(pd.DataFrame.from_dict(score_kmean_2, orient="index", columns=["Valor"]))

        with col2:
            st.markdown("### Gaussian Mixture")

            agrupamiento_gmm_2 = graph_3d(
                    datos_pca,
                    "Gaussian Mixture",
                    y_hat_gmm_2,
                    columns_select[0],
                    columns_select[1],
                    columns_select[2],
                    *pca_columns
                    )
            st.pyplot(agrupamiento_gmm_2)

            st.dataframe(pd.DataFrame.from_dict(score_gmm_2, orient="index", columns=["Valor"]))

        with col3:    
            st.markdown("### DBSCAN")

            agrupamiento_dbscan_2 = graph_3d(
                    datos_pca,
                     "DBSCAN",
                    y_hat_dbscan_2,
                    columns_select[0],
                    columns_select[1],
                    columns_select[2],
                    *pca_columns
                    )

            st.pyplot(agrupamiento_dbscan_2)

            st.dataframe(pd.DataFrame.from_dict(score_dbscan_2, orient="index", columns=["Valor"]))

        ####################################################
        st.subheader("Elección del Modelo 2")

        st.markdown("Basados en los scores anteriores tenemos que el mejor modelo en este caso es:")
        modelos_2 = st.selectbox("Selecciona el modelo (PCA):" , ["Kmeans","Gaussian Mixture", "DBSCAN"])
        if modelos_2:
            if modelos_2 == "Kmeans":
                y_hat_modelo_2 = y_hat_kmean_2
                score_modelo_2 = score_kmean_2
                modelo_2 = km_2
                columns_select_modelo_2 = pca_columns
            elif modelos_1 == "Gaussian Mixture":
                y_hat_modelo_2 = y_hat_gmm_2
                score_modelo_2 = score_gmm_2
                modelo_2 = gmm_2
                columns_select_modelo_2 = pca_columns
            else:
                y_hat_modelo_2 = y_hat_dbscan_2
                score_modelo_2 = score_dbscan_2
                modelo_2 = dbscan_model_2
                columns_select_modelo_2 = pca_columns

        ####################################################
        ####################################################
        ####################################################
        ####################################################
        ####################################################

        st.markdown("<hr>", unsafe_allow_html=True)

        st.subheader("Modelo Final")

        col1, col2 = st.columns([1,1])

        with col1:
            st.markdown("### Modelo 1")
            st.dataframe(pd.DataFrame.from_dict(score_modelo_1, orient="index", columns=["Valor"]))

        with col2:
            st.markdown("### Modelo 2")
            st.dataframe(pd.DataFrame.from_dict(score_modelo_2, orient="index", columns=["Valor"]))


        st.markdown("Basados en los scores anteriores tenemos que el mejor modelo en este caso es:")
        modelos_finales = st.selectbox("Selecciona modelo final :" , ["Modelo 1","Modelo 2"])

        if modelos_finales:
            if modelos_finales == "Modelo 1":
                st.session_state.y_hat_modelo_final = y_hat_modelo_1
                st.session_state.score_modelo_final = score_modelo_1
                st.session_state.modelo_final = modelo_1
                st.session_state.columns_select_modelo_final = columns_select
            else:
                st.session_state.y_hat_modelo_final = y_hat_modelo_2
                st.session_state.score_modelo_final = score_modelo_2            
                st.session_state.modelo_final = modelo_2
                st.session_state.columns_select_modelo_final = pca_columns

        ####################################################
        ####################################################
        ####################################################
        ####################################################
        ####################################################

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Clasificación de los Segmentos")
        st.write("Veamos las características de los distintos clusters (definidos por el modelo final) para terminar la clasificación de los clientes:")

        # Agrego la clasificación usando el mejor modelo
        st.session_state.rfm['Cluster'] = st.session_state.y_hat_modelo_final
        st.session_state.rfm_std['Cluster'] = st.session_state.y_hat_modelo_final
        st.session_state.datos_etiquetados_final = pd.merge(st.session_state.datos, st.session_state.rfm.loc[:,["CustomerNo", "Cluster"]], how="left", on="CustomerNo")

        # Calcular el análisis por cluster en el rfm
        cluster_analysis = st.session_state.rfm.groupby('Cluster')[st.session_state.columns_select_modelo_final].mean()
        scaler = MinMaxScaler()

        cluster_analysis_normalized = pd.DataFrame(
            scaler.fit_transform(cluster_analysis),
            index=cluster_analysis.index,
            columns=cluster_analysis.columns
        )

        bargraph = bar_graph(
            cluster_analysis_normalized,
            "Estadísticas promedio por cluster",
            "Métricas",
            "Clusters",
            "Valores promedio",
            *st.session_state.columns_select_modelo_final
            )

        col1, col2,col3 = st.columns([1,2,1])
        with col2:
            st.pyplot(bargraph)

        st.write("Clasifiquemos los clusters encontrados:")
        # esto es por inspección

        # se puede automatizar pidiendo un input

        n = len(set(st.session_state.y_hat_modelo_final))  # Número de columnas deseadas (puedes cambiarlo dinámicamente)
        columnas = st.columns([1] * n) 

        st.session_state.clusters_labels_final = {}

        clusters_labels_inputs = []
        for i in range(n):
            with columnas[i]:
                st.session_state.clusters_labels_final[i]  = st.text_area(f"Clasificación para {i}: ")





        st.markdown(
                    """
                    En este caso hemos obtenido 4 clusters, y viendo los promedios de las tres componentes, podemos clasificarlos como:
                    - **Excelente (Cluster 3)**: Baja recency, alta frecuencia, alto valor monetario.
                    - **Bueno (Cluster 2)**: Media recency, alta frecuencia, alto valor monetario.
                    - **Regular (Cluster 0)**: Media recency, media/baja frecuencia, media/baja valor monetario.
                    - **Inactivo (Cluster 1)**: Alta recency, baja frecuencia, bajo valor monetario.
                    """
                    )

    

        agrupamiento_final = graph_3d_segements(st.session_state.rfm_std,
                                                st.session_state.y_hat_modelo_final,
                                                st.session_state.columns_select_modelo_final[0],
                                                st.session_state.columns_select_modelo_final[1],
                                                st.session_state.columns_select_modelo_final[2],
                                                st.session_state.clusters_labels_final,
                                                *st.session_state.columns_select_modelo_final
                                                )

        col1, col2,col3 = st.columns([1,2,1])
        with col2:
            st.pyplot(agrupamiento_final)
    else:
        st.warning("Por favor, genere el RFM dataframe.")