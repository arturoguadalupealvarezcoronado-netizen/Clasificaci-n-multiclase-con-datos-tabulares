# librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


# cargar dataset
dataset = fetch_ucirepo(name='Wine')

X = dataset.data.features
y = dataset.data.targets

print("Muestras:", X.shape[0])
print("Variables:", X.shape[1])
print("Clases:", np.unique(y))


# convertir clases
le = LabelEncoder()
y = le.fit_transform(y.values.ravel())

y_cat = tf.keras.utils.to_categorical(y)


# dividir datos
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_cat, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


# escalar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# modelos
model1 = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model2 = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model3 = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])


# compilar
def compile_model(model):
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

for model in [model1, model2, model3]:
    compile_model(model)


# entrenar
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

histories = []

for model in [model1, model2, model3]:
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        callbacks=[early_stop],
        verbose=1
    )
    histories.append(history)


# gráficas
plt.figure(figsize=(12,5))

# loss
plt.subplot(1,2,1)
for history in histories:
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
plt.title("Loss")

# accuracy
plt.subplot(1,2,2)
for history in histories:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")

plt.show()


# evaluación
for i, model in enumerate([model1, model2, model3]):
    print(f"\nModelo {i+1}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(confusion_matrix(y_true, y_pred_classes))
    print(classification_report(y_true, y_pred_classes))

    print("ROC-AUC:", roc_auc_score(y_test, y_pred, multi_class='ovr'))
    print("PR-AUC:", average_precision_score(y_test, y_pred))

    sns.heatmap(confusion_matrix(y_true, y_pred_classes), annot=True)
    plt.title(f"Modelo {i+1}")
    plt.show()


# precision-recall
plt.figure()

for model in [model1, model2, model3]:
    y_pred = model.predict(X_test)

    for j in range(3):
        precision, recall, _ = precision_recall_curve(y_test[:, j], y_pred[:, j])
        plt.plot(recall, precision)

plt.title("PR Curve")
plt.show()


# distribución de clases
unique, counts = np.unique(y, return_counts=True)

plt.bar(unique, counts)
plt.title("Clases")
plt.show()


# umbral
for i, model in enumerate([model1, model2, model3]):
    y_pred = model.predict(X_test)

    confidence = np.max(y_pred, axis=1)
    predictions = np.argmax(y_pred, axis=1)
    true = np.argmax(y_test, axis=1)

    mask = confidence > 0.7

    print(f"\nModelo {i+1} umbral 0.7:")
    print("Accuracy:", np.mean(predictions[mask] == true[mask]))