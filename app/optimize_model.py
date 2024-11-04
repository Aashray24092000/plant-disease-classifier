import tensorflow as tf

# Load your existing model
model_path = r'C:\Users\hp\OneDrive\Desktop\plant-disease-image-prediction-master\app\plant_disease_prediction_model.h5'  # Use the full path
 # Update this path
model = tf.keras.models.load_model(model_path)

# Convert the model to TensorFlow Lite format with optimizations
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Use default optimization (including quantization)

# Convert the model
tflite_model = converter.convert()

# Save the optimized model to a file
tflite_model_path = 'optimized_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'Model has been converted and saved to {tflite_model_path}')
