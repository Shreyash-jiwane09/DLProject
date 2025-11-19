from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)




@app.route("/health", methods=['GET'])
@cross_origin()
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Lung Cancer Classification API",
        "version": "1.0.0"
    })


@app.route("/model/info", methods=['GET'])
@cross_origin()
def model_info():
    """Get model information"""
    try:
        model_path = os.path.join("artifacts", "training", "model.h5")
        metrics_path = "metrics.json"
        
        info = {
            "model_architecture": "VGG16 Transfer Learning",
            "classes": ["adenocarcinoma", "normal"],
            "input_size": [224, 224, 3],
            "model_exists": os.path.exists(model_path)
        }
        
        # Add metrics if available
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                info["accuracy"] = metrics.get("accuracy", "N/A")
                info["roc_auc_score"] = metrics.get("roc_auc_score", "N/A")
        
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({"error": "Failed to get model info"}), 500


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')



@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    try:
        logger.info("Starting training via DVC...")
        #os.system("python main.py")
        exit_code = os.system("dvc repro")
        
        if exit_code == 0:
            return jsonify({"message": "Training completed successfully!"}), 200
        else:
            return jsonify({"error": "Training failed. Check logs for details."}), 500
    except Exception as e:
        logger.error(f"Error in training route: {str(e)}")
        return jsonify({"error": "Training failed"}), 500


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        # Validate request
        if not request.json or 'image' not in request.json:
            return jsonify({"error": "No image data provided"}), 400
        
        image = request.json['image']
        filename = "inputImage.jpg"
        
        # Decode and save image
        decodeImage(image, filename)
        
        # Validate file was created
        if not os.path.exists(filename):
            return jsonify({"error": "Failed to decode image"}), 400

        # Create a new instance with the updated filename
        predictor = PredictionPipeline(filename)
        result = predictor.predict()
        
        # Check for errors in prediction
        if result and "error" in result[0]:
            return jsonify(result[0]), 400
        
        return jsonify(result[0])
    
    except Exception as e:
        logger.error(f"Error in prediction route: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500



if __name__ == "__main__":
    clApp = ClientApp()
    logger.info("Starting Flask application...")
    logger.info("Server running at http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)






