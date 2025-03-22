# backend/app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from model.chatbot import WasteManagementChatbot
from model.classify import WasteImageClassifier

app = Flask(__name__)
# Allow cross-origin requests from any origin
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize models
waste_chatbot = WasteManagementChatbot()
waste_classifier = WasteImageClassifier()

@app.route('/', methods=['GET'])
def home():
    """Basic home route for testing."""
    return jsonify({
        'status': 'online',
        'message': 'Waste Management Assistant API is running',
        'endpoints': [
            '/api/health',
            '/api/waste-management-advice',
            '/api/classify-waste-image'
        ]
    })

@app.route('/api/waste-management-advice', methods=['POST'])
def get_waste_management_advice():
    """Endpoint to get waste management advice for an item or question."""
    if not request.json or 'userInput' not in request.json:
        return jsonify({'error': 'Missing userInput parameter'}), 400
    
    user_input = request.json['userInput']
    advice = waste_chatbot.generate_waste_advice(user_input)
    return jsonify(advice)

@app.route('/api/classify-waste-image', methods=['POST'])
def classify_waste_image():
    """Endpoint to classify waste from an uploaded image."""
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'Missing image file'}), 400
            
        file = request.files['image']
        
        # Get image classification
        classification_result = waste_classifier.predict_waste_type(file)
        
        # Get detailed advice for the predicted waste type
        waste_type = classification_result["predicted_class"]
        advice = waste_chatbot.generate_waste_advice(waste_type)
        
        # Combine the results
        result = {
            "classification": classification_result,
            "advice": advice
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in classification endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'healthy', 
        'service': 'waste-management-assistant',
        'version': '1.0',
        'components': {
            'chatbot': 'active',
            'image_classifier': 'active'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    # Set host to 0.0.0.0 to make it accessible from other machines on the network
    app.run(host='0.0.0.0', port=port, debug=True)