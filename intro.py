import streamlit as st

def intro_page():
    st.header("Página: Descripción del Proyecto")

   # col1, col2 , col3 = st.columns([1,8,1])
   # with col2:
   #     st.image("BMC.png")
   #     st.image("MLC.png")
   #
   # # Leer el archivo markdown
   # with open("flujo_proyecto.md", "r") as file:
   #     contenido_markdown = file.read()
   #
   # # Mostrar el contenido del archivo markdown en Streamlit
   # st.markdown("---")
   # st.markdown(contenido_markdown)
    st.markdown(
        """
        ### Proyecto: Optimización de Gestión para Tienda Online  

        #### Contexto  
        Un emprendedor con una tienda online en **WhatsApp** especializada en **zapatos y accesorios** experimentó un crecimiento en su negocio.  

        #### Problema  
        La falta de una **estructura sólida de registro de compras** dificultaba el **análisis del negocio** y el **diseño de estrategias de crecimiento**.  

        #### Objetivo del Proyecto  
        - Implementar un **sistema de gestión de compras** para mejorar el **seguimiento de pedidos y análisis de ventas**.  
        - Facilitar la toma de decisiones basada en **datos estructurados**.  
        """
        )
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.image("BMC_V.png")
    st.markdown("---")
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.image("MLC_V.png")
    st.markdown("---")
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.image("FLUJO_V.png")