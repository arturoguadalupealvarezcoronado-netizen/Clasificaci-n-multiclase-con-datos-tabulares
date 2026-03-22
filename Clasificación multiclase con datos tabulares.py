import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


# Carga del dataset
data = load_iris()
X = data.data
y = data.target

print("Muestras:", X.shape[0])
print("Variables:", X.shape[1])
print("Clases:", len(np.unique(y)))


# Preprocesamiento de datos
y_cat = tf.keras.utils.to_categorical(y)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_cat, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# Definición de modelos
model1 = Sequential([
    Input(shape=(4,)),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model2 = Sequential([
    Input(shape=(4,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model3 = Sequential([
    Input(shape=(4,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])


# Compilación de modelos
def compile_model(model):
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

for model in [model1, model2, model3]:
    compile_model(model)


# Entrenamiento de modelos
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

histories = []

for model in [model1, model2, model3]:
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        callbacks=[early_stop],
        verbose=0
    )
    histories.append(history)


# Visualización de desempeño
fig, axs = plt.subplots(1, 3, figsize=(18,5))
colores = ['blue', 'green', 'red']


# Curvas de pérdida
for i, history in enumerate(histories):
    axs[0].plot(history.history['loss'], color=colores[i], label=f'Modelo {i+1} Entrenamiento')
    axs[0].plot(history.history['val_loss'], linestyle='--', color=colores[i], label=f'Modelo {i+1} Validación')

axs[0].set_title("Curvas de pérdida")
axs[0].set_xlabel("Épocas")
axs[0].set_ylabel("Loss")
axs[0].legend()


# Curvas de exactitud
for i, history in enumerate(histories):
    axs[1].plot(history.history['accuracy'], color=colores[i], label=f'Modelo {i+1} Entrenamiento')
    axs[1].plot(history.history['val_accuracy'], linestyle='--', color=colores[i], label=f'Modelo {i+1} Validación')

axs[1].set_title("Curvas de exactitud")
axs[1].set_xlabel("Épocas")
axs[1].set_ylabel("Accuracy")
axs[1].legend()


# Curvas ROC multiclase
y_true = np.argmax(y_test, axis=1)
y_test_bin = label_binarize(y_true, classes=[0,1,2])

for i, model in enumerate([model1, model2, model3]):
    y_pred = model.predict(X_test)

    for j in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, j], y_pred[:, j])
        axs[2].plot(fpr, tpr, label=f'M{i+1} Clase {j}')

axs[2].set_title("Curvas ROC multiclase")
axs[2].set_xlabel("FPR")
axs[2].set_ylabel("TPR")
axs[2].legend()


plt.tight_layout()
plt.show()


# Evaluación y matrices de confusión
for i, model in enumerate([model1, model2, model3]):
    print(f"\nModelo {i+1}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred_classes))

    print("\nReporte de clasificación:")
    print(classification_report(y_true, y_pred_classes))

    print("ROC-AUC:", roc_auc_score(y_test, y_pred, multi_class='ovr'))
    print("PR-AUC:", average_precision_score(y_test, y_pred))

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de confusión - Modelo {i+1}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()


# Curvas Precision-Recall
plt.figure()

for i, model in enumerate([model1, model2, model3]):
    y_pred = model.predict(X_test)

    for j in range(3):
        precision, recall, _ = precision_recall_curve(y_test[:, j], y_pred[:, j])
        plt.plot(recall, precision, label=f'M{i+1} Clase {j}')

plt.title("Curvas Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()


# Distribución de clases
unique, counts = np.unique(y, return_counts=True)

plt.figure()
plt.bar(unique, counts)
plt.title("Distribución de clases")
plt.xlabel("Clase")
plt.ylabel("Cantidad")
plt.show()