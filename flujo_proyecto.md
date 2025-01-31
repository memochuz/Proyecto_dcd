## Flujo de trabajo
### KDD (Knowledge Discovery in Databases)

#### 1. Selección de Datos
Elegimos los datos relevantes para el análisis:

* Ramos, G. (s.f.). An online shop business. Kaggle. Recuperado el 30 de enero de 2025, de https://www.kaggle.com/datasets/gabrielramos87/an-online-shop-business

* Estos datos fueron escogidos por mantener una estructura mínima que un negocio de e-commerce debería tener.

#### 2. Preprocesamiento (Limpieza)
Preparamos los datos para que sean adecuados para el análisis:

- Se quitaron registros con valores faltantes en `CustomerNo`, ya que no había forma de relacionarlos con alguna transacción.  

- Se verificó que los números negativos en la columna `Quantity` se debían a cancelaciones. Estas también fueron eliminadas, ya que, por el momento, no proporcionan información relevante para nuestro análisis.  

- Convertimos las strings en la columna `ProductName` a minúsculas para evitar errores por duplicados.  

- Añadimos una columna llamada `TotalSpent`, que incluye el precio total de la compra.  

- Transformamos la columna `Date` al formato correcto usando `pd.to_datetime`.  

- Identificamos registros sin país, pero al contar con su número de cliente, los dejamos para un primer análisis en el que no segmentaremos por región.  

- Verificamos que no haya problemas con los nombres de los productos usando el siguiente patrón: buscamos strings que incluyan signos de interrogación y tengan una longitud menor a 4, lo que indicaría un problema con el dato.  

- Observamos que la columna `ProductNo` no es de tipo entero porque algunos productos tienen códigos que incluyen letras.  

#### 3. Transformación
Transformamos los datos según las necesidades del análisis:  

* Construimos las tablas RFM y RFM estandarizada. Esta gráfica se obtiene calculando las siguientes características:  

    | Métrica                  | Descripción |
    |--------------------------|------------|
    | **Recency (R)**          | Mide el tiempo transcurrido desde la última compra de un cliente. Un valor bajo indica que el cliente compró recientemente, lo que sugiere mayor compromiso. |
    | **Frequency (F)**        | Indica el número de compras realizadas en un período determinado. Un cliente con una alta frecuencia es más leal y activo. |
    | **Monetary (M)**         | Representa el monto total gastado por el cliente en un período determinado. Clientes con un alto valor monetario son más valiosos para el negocio. |
    | **AverageBetweenPurchases** | Representa el número promedio de días que un cliente tarda en volver a realizar una compra. |
    | **AvgPurchaseSize**      | Representa el gasto promedio por compra. |

* Generamos una matriz de transacciones-productos para utilizar el método Apriori. Esta matriz se crea a partir de la tabla obtenida después del proceso de limpieza de datos.

#### 4. Minería de Datos  
Aquí realizamos el análisis principal:  

* Utilizamos la tabla RFM estandarizada para realizar las agrupaciones con los algoritmos **K-Means**, **DBSCAN** y **Gaussian Mixture Model (GMM)**.  

* Se creó un modelo en el que se pueden seleccionar distintas características de la tabla para realizar las clusterizaciones. Se desarrollarán dos modelos: uno utilizando solo tres características principales y otro considerando las cinco en total.  

    Estas clusterizaciones se obtienen comparando los tres métodos de agrupación mencionados anteriormente y seleccionando el más adecuado.  

    Una vez obtenidos los dos modelos, se aplicará nuevamente un proceso de eliminación para escoger uno de ellos y determinar el modelo final.  

* Para **K-Means** y **GMM**, se utilizó una gráfica de codo para determinar el número óptimo de clústeres.  
* En el caso de **DBSCAN**, se usó una búsqueda en malla (*grid search*) para optimizar los hiperparámetros clave: **distancia épsilon (`eps`)** y **número mínimo de vecinos (`min_samples`)**.  


* Luego, aplicamos el algoritmo **Apriori** para identificar productos frecuentemente comprados juntos en cada clúster.  

* Finalmente, examinamos las **tendencias estacionales** en las transacciones para determinar estrategias basadas en los resultados obtenidos.  


#### 5. Evaluación  

Evaluamos los patrones descubiertos para asegurarnos de que sean útiles:  

* Validamos las reglas de asociación con métricas como **soporte**, **confianza** y **lift**.  
* Evaluamos la calidad de los clústeres mediante métricas como **Silhouette Score** y **Calinski-Harabasz Score**. 
    - **Silhouette Score**: Evalúa la separación y cohesión de los clústeres.  
    - **Calinski-Harabasz Score**: Mide la relación entre la dispersión dentro y entre los clústeres.  

  Estas métricas nos ayudaron a seleccionar la mejor clusterización.  
* Identificamos si las reglas descubiertas son **prácticas y accionables** para su implementación.

#### 6. Presentación del Conocimiento  

Presentamos los resultados de forma clara:  

* Resúmenes de las **reglas de asociación más fuertes**.  
* **Visualizaciones** de los clústeres y reglas de clasificación.  
* **Propuestas de acción** basadas en los descubrimientos obtenidos en el análisis.  

