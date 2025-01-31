import streamlit as st

def intro_page():
    st.header("Página: Descripción del Proyecto")

    col1, col2 , col3 = st.columns([1,8,1])
    with col2:
        st.image("BMC.png")
        st.image("MLC.png")

    # Leer el archivo markdown
    with open("/home/memochuz/Documents/Notas_de_mis_cursos/Ciencia_de_Datos_UNAM/Proyecto/Documentos/flujo_proyecto.md", "r") as file:
        contenido_markdown = file.read()

    # Mostrar el contenido del archivo markdown en Streamlit
    st.markdown("---")
    st.markdown(contenido_markdown)