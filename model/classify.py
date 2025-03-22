from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
import warnings
# import tensorflow as tf

class WasteImageClassifier:
    def __init__(self, model_path='waste_classifier_model.h5'):
        """Initialize waste image classifier with TensorFlow model if available,
        otherwise use color-based heuristics as fallback."""
        self.class_labels = ['paper', 'trash', 'plastic', 'metal', 'biological']
        self.model = None
        self.using_ml_model = False
        
        # Try to load TensorFlow and the ML model
        # try:
        #     self.model = tf.keras.models.load_model(model_path)
        #     print("TensorFlow model loaded successfully!")
        #     print(self.model.summary())
        #     self.using_ml_model = True
        # except Exception as e:
        #     warnings.warn(f"Could not load TensorFlow model: {e}")
        #     print("Falling back to color-based heuristic classifier")
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction."""
        img = cv2.resize(image, (128, 128))  # Resize to match training input size
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    
    def predict_with_ml_model(self, image):
        """Predict waste type using TensorFlow model."""
        preprocessed_img = self.preprocess_image(image)
        predictions = self.model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = self.class_labels[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Get confidence scores for all classes
        all_confidences = {
            self.class_labels[i]: float(predictions[0][i]) 
            for i in range(len(self.class_labels))
        }
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_confidences": all_confidences,
            "method": "machine_learning"
        }
    
    def predict_with_heuristic(self, image):
        """Predict waste type using color-based heuristics."""
        # Extract image features (using color distribution as a simple heuristic)
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_img], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Simple heuristic: classify based on dominant color
        if hist[0] > 0.5:
            predicted_class = 'paper'
        elif hist[1] > 0.5:
            predicted_class = 'plastic'
        elif hist[2] > 0.5:
            predicted_class = 'metal'
        elif hist[3] > 0.5:
            predicted_class = 'biological'
        else:
            predicted_class = 'trash'
        
        confidence = 0.8  # This is a heuristic, so we use a reasonable confidence value
        
        # Get confidence scores for all classes (dummy values for heuristic)
        class_confidences = {label: 0.2 for label in self.class_labels}
        class_confidences[predicted_class] = confidence
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_confidences": class_confidences,
            "method": "color_heuristic"
        }
    
    def predict_waste_type(self, image_file):
        """
        Classify waste type from an image file.
        
        Args:
            image_file: An image file object or bytes
            
        Returns:
            Dictionary with predicted class and confidence score
        """
        try:
            # Check if input is a file-like object or bytes
            if hasattr(image_file, 'read'):
                # It's a file-like object
                img_bytes = image_file.read()
            else:
                # Assume it's already bytes
                img_bytes = image_file
            
            # Decode the image
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            # Use ML model if available, otherwise use heuristic
            if self.using_ml_model:
                try:
                    return self.predict_with_ml_model(img)
                except Exception as e:
                    print(f"ML model prediction failed: {e}. Falling back to heuristic.")
                    return self.predict_with_heuristic(img)
            else:
                return self.predict_with_heuristic(img)
            
        except Exception as e:
            print(f"Error in image classification: {e}")
            # Return a default response in case of error
            return {
                "predicted_class": "trash",
                "confidence": 0.5,
                "all_confidences": {label: 0.2 for label in self.class_labels},
                "error": str(e),
                "method": "fallback"
            }

# Initialize Flask app
app = Flask(__name__)
classifier = WasteImageClassifier()

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read image from request
        file = request.files['image']
        
        # Make prediction
        result = classifier.predict_waste_type(file)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
