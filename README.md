# Clasificación Multiclase con MLP

# Descripción
Este proyecto implementa y evalúa tres modelos de redes neuronales densas (MLP) para clasificación multiclase utilizando TensorFlow/Keras.

# Dataset
Se utilizó el dataset Iris, el cual contiene:
- 150 muestras
- 4 variables
- 3 clases

# Preprocesamiento
- División de datos: entrenamiento (70%), validación (15%) y prueba (15%)
- Escalamiento de variables con StandardScaler
- Codificación de etiquetas mediante one-hot encoding

# Modelos
Se implementaron tres modelos con diferentes arquitecturas:

# Modelo 1
- 1 capa oculta (16 neuronas)
- Activación ReLU

# Modelo 2
- 2 capas ocultas (32 y 16 neuronas)
- Activación ReLU

# Modelo 3
- 3 capas ocultas (64, 32 y 16 neuronas)
- Activación ReLU

Todos los modelos utilizan:
- Función de pérdida: categorical_crossentropy
- Optimizador: Adam
- Capa de salida con función softmax

# Evaluación
Se utilizaron las siguientes métricas:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC multiclase
- PR-AUC multiclase

# Resultados
Los modelos fueron evaluados en el conjunto de prueba. El modelo más complejo mostró mejor desempeño general, mientras que el modelo intermedio presentó un buen balance entre precisión y complejidad.

# Archivos
- proyecto_mlp.py → Código principal
- README.md → Descripción del proyecto

# Autor
Arturo Guadalupe Alvarez Coronado