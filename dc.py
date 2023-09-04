import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

(x_train_digits, y_train_digits), (x_test_digits, y_test_digits) = mnist.load_data()
x_train_digits, x_test_digits = x_train_digits / 255.0, x_test_digits / 255.0  

num_custom_samples = 1000
custom_characters = np.random.rand(num_custom_samples, 28, 28)
custom_labels = np.random.choice(26, num_custom_samples)  # Assuming 26 characters

x_train_combined = np.concatenate((x_train_digits, custom_characters), axis=0)
y_train_combined = np.concatenate((y_train_digits, custom_labels), axis=0)

label_encoder = LabelEncoder()
y_train_combined_encoded = label_encoder.fit_transform(y_train_combined)
num_classes = len(label_encoder.classes_)

x_train, x_val, y_train, y_val = train_test_split(x_train_combined, y_train_combined_encoded, test_size=0.2, random_state=42)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

test_loss_digits, test_accuracy_digits = model.evaluate(x_test_digits, y_test_digits, verbose=2)
print(f"Test accuracy for MNIST digits: {test_accuracy_digits}")

test_loss_custom, test_accuracy_custom = model.evaluate(custom_characters, custom_labels, verbose=2)
print(f"Test accuracy for custom characters: {test_accuracy_custom}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. Epochs')

plt.show()
