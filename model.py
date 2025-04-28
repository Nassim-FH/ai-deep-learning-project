import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_data():
    """Load and preprocess MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    return (x_train, y_train), (x_test, y_test)

def build_model(input_shape=(28, 28, 1)):
    """Build CNN model"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    return model

def plot_history(history):
    """Plot training history"""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')  # Save plot
    plt.show()

def main():
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Build model
    model = build_model()
    model.summary()  # Show model architecture
    
    # Train model
    history = model.fit(x_train, y_train, 
                        epochs=5, 
                        batch_size=64,
                        validation_split=0.1)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Save model and plots
    model.save('mnist_cnn_model.h5')
    plot_history(history)

if __name__ == "__main__":
    main()