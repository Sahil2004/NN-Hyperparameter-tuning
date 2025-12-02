from flask import Flask, render_template, request, jsonify
import os
import json
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import after dotenv is loaded
try:
    from crew_ai_agents import create_crew, process_dataset_request
except ValueError as e:
    # Handle missing API key gracefully
    print(f"Warning: {e}")
    process_dataset_request = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    """Main page with dataset upload/selection interface"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Process dataset request and run CrewAI agents"""
    if process_dataset_request is None:
        return jsonify({
            'success': False,
            'error': 'Google API key not configured. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file. Get a free key from https://aistudio.google.com/u/1/api-keys'
        }), 500
    
    try:
        data = request.json
        dataset_type = data.get('dataset_type')  # 'upload' or 'mnist' or 'cifar'
        dataset_summary = data.get('dataset_summary', '')
        dataset_file = None
        
        if dataset_type == 'upload':
            # Handle file upload if provided
            if 'file' in request.files:
                file = request.files['file']
                if file.filename:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(filepath)
                    dataset_file = filepath
        
        # Create context for agents
        context = {
            'dataset_type': dataset_type,
            'dataset_summary': dataset_summary or get_default_dataset_info(dataset_type),
            'dataset_file': dataset_file
        }
        
        # Run CrewAI agents
        result = process_dataset_request(context)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        error_details = traceback.format_exc()
        error_message = str(e)
        
        # Handle specific Google API errors
        if '429' in error_message or 'RESOURCE_EXHAUSTED' in error_message or 'quota' in error_message.lower():
            error_message = (
                "You've exceeded your free tier quota or rate limit. "
                "Please wait a few minutes before trying again. "
                "Free tier limits: 15 requests per minute (RPM) and 1 million tokens per minute (TPM). "
                "For more info: https://ai.google.dev/gemini-api/docs/rate-limits"
            )
        elif '404' in error_message or 'NOT_FOUND' in error_message:
            error_message = (
                "Model not found or not available in your API access level. "
                "Common issues:\n"
                "1. The model name may not be available on your free tier\n"
                "2. Check available models at: https://ai.google.dev/gemini-api/docs/models\n"
                "3. Try using 'gemini-pro' in crew_ai_agents.py (line 24)\n"
                "4. Verify your API key is correct and active"
            )
        elif '401' in error_message or 'UNAUTHENTICATED' in error_message:
            error_message = (
                "Authentication failed. Please check your GOOGLE_API_KEY in the .env file. "
                "Get a free key from: https://aistudio.google.com/u/1/api-keys"
            )
        
        print(f"Error in /api/analyze: {error_details}")
        return jsonify({
            'success': False,
            'error': error_message,
            'details': error_details if app.debug else None
        }), 500


def get_default_dataset_info(dataset_type):
    """Get default dataset information for MNIST or CIFAR"""
    datasets = {
        'mnist': """
        Dataset: MNIST (Modified National Institute of Standards and Technology)
        Type: Image Classification
        Classes: 10 (digits 0-9)
        Image Size: 28x28 grayscale
        Training Samples: 60,000
        Test Samples: 10,000
        Task: Handwritten digit recognition
        """,
        'cifar': """
        Dataset: CIFAR-10
        Type: Image Classification
        Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
        Image Size: 32x32 RGB color images
        Training Samples: 50,000
        Test Samples: 10,000
        Task: Object recognition in natural images
        """
    }
    return datasets.get(dataset_type, '')


if __name__ == '__main__':
    app.run(debug=True, port=5000)

