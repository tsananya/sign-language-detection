import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # Import load_model
import cv2

# Step 1: Load the Pre-trained Model
# This line loads the model you trained and saved in the previous step
model = load_model('sign_language_model.h5')

# Step 2: Convert to TensorFlow Lite Format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = "sign_language_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
print(f"Model saved to {tflite_model_path}")

# Step 3: Set Up TensorFlow Lite Interpreter for Real-time Detection
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Step 4: Real-time Gesture Recognition using OpenCV
cap = cv2.VideoCapture(0) # Open the webcam
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Preprocess the frame for the model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    input_data = np.expand_dims(resized, axis=(0, -1)) / 255.0

    # Set the input tensor and invoke the model
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class
    predicted_class = np.argmax(output_data)

    # Display the prediction on the video frame
    label_text = f"Predicted: {chr(predicted_class + ord('A'))}"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Sign Language Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()