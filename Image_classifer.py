import numpy as np
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Define data directory path
DATA_DIR = os.path.join(os.getcwd(), 'data')  # Create a 'data' folder in current directory

# Function to load data from specified directory
def load_data(data_dir):
    """Load training and test data from specified directory"""
    # Load training data
    train_input_path = os.path.join(data_dir, 'train', 'input.csv')
    train_labels_path = os.path.join(data_dir, 'train', 'labels.csv')
    X_train = np.loadtxt(train_input_path, delimiter=',')  # Fixed typo in delimiter
    Y_train = np.loadtxt(train_labels_path, delimiter=',')  # Fixed typo in delimiter
    
    # Load test data
    test_input_path = os.path.join(data_dir, 'test', 'input_test.csv')
    test_labels_path = os.path.join(data_dir, 'test', 'labels_test.csv')
    X_test = np.loadtxt(test_input_path, delimiter=',')
    Y_test = np.loadtxt(test_labels_path, delimiter=',')
    
    return X_train, Y_train, X_test, Y_test

# Load data from directory
X_train, Y_train, X_test, Y_test = load_data(DATA_DIR)

# Print shapes of loaded data for verification
print("Shape of X_train:", X_train.shape)
print("Shape of Y_train:", Y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of Y_test:", Y_test.shape)  # Fixed variable name

# Reshape input data to match CNN requirements (samples, height, width, channels)
X_train = X_train.reshape(len(X_train), 100, 100, 3)
X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)  # Fixed from X_train to Y_train
Y_test = Y_test.reshape(len(Y_test), 1)     # Fixed from X_test to Y_test

# Display a random training image
def show_random_image(X_data, title="Random Image"):
    """Display a random image from the dataset"""
    idx = random.randint(0, len(X_data)-1)
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(X_data[idx])
    plt.axis('off')
    plt.show()
    return idx

# Show random training image
show_random_image(X_train, "Random Training Image")

# Build CNN model
def create_model():
    """Create and return the CNN model"""
    model = Sequential()
    # First convolutional layer with 32 filters
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Second convolutional layer with 64 filters
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the output for dense layers
    model.add(Flatten())
    # Two dense layers for classification
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Fixed to 1 unit for binary classification
    
    return model

# Create and compile model
model = create_model()
optimizer = SGD(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate model on test data
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Function to make prediction on a single image
def predict_image(model, image, classes=['Dog', 'Cat']):
    """Make prediction on a single image"""
    prediction = model.predict(image.reshape(1, 100, 100, 3))
    class_idx = int(prediction > 0.5)
    return classes[class_idx], prediction[0][0]

# Show and predict random test image
test_idx = show_random_image(X_test, "Random Test Image")
predicted_class, confidence = predict_image(model, X_test[test_idx])
print(f"Model predicts this is a {predicted_class} with confidence: {confidence:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()