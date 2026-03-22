import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
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


# Curvas de pérdida
plt.figure()

for i, history in enumerate(histories):
    plt.plot(history.history['loss'], label=f'Modelo {i+1} Entrenamiento')
    plt.plot(history.history['val_loss'], linestyle='--', label=f'Modelo {i+1} Validación')

plt.title("Curvas de pérdida")
plt.legend()
plt.show()


# Curvas de exactitud
plt.figure()

for i, history in enumerate(histories):
    plt.plot(history.history['accuracy'], label=f'Modelo {i+1} Entrenamiento')
    plt.plot(history.history['val_accuracy'], linestyle='--', label=f'Modelo {i+1} Validación')

plt.title("Curvas de exactitud")
plt.legend()
plt.show()


# Evaluación de modelos
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


# Curvas ROC multiclase
from sklearn.metrics import roc_curve

y_true = np.argmax(y_test, axis=1)
y_test_bin = label_binarize(y_true, classes=[0, 1, 2])

plt.figure()

for i, model in enumerate([model1, model2, model3]):
    y_pred = model.predict(X_test)

    for j in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, j], y_pred[:, j])
        plt.plot(fpr, tpr, label=f'Modelo {i+1} Clase {j}')

plt.title("Curvas ROC multiclase")
plt.legend()
plt.show()