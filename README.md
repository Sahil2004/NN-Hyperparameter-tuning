# Neural Network Hyperparameter Tuner

An AI-powered Flask application that uses a CrewAI crew of 4 specialized agents to collaboratively design optimal neural network architectures, recommend hyperparameters, and generate ready-to-run PyTorch training scripts.

## Features

- **Dataset Selection**: Choose from MNIST, CIFAR-10, or upload a custom dataset summary
- **4 AI Agents Working in Harmony**:
  1. **Data Analyst Agent**: Analyzes dataset characteristics and constraints
  2. **Model Designer Agent**: Proposes optimal neural network architecture (MLP/CNN/RNN)
  3. **Hyperparameter Agent**: Suggests learning rate, optimizer, epochs, and other hyperparameters
  4. **CodeGen Agent**: Generates complete PyTorch training scripts

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd nn-hyperparameter-tuner
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Google Gemini API key:

   Get a free API key from [Google AI Studio](https://aistudio.google.com/u/1/api-keys)
   
   Create a `.env` file in the project root and add your API key:
   ```
   GOOGLE_API_KEY=your-api-key-here
   ```
   
   Or set it as an environment variable:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   # Alternative: export GEMINI_API_KEY="your-api-key-here"
   ```
   
   **Free Tier Limits:**
   - 15 requests per minute (RPM)
   - 1 million tokens per minute (TPM)
   - If you hit rate limits, wait a few minutes before trying again

5. (Optional) Verify your setup:
   ```bash
   python check_setup.py
   ```

6. (Optional) Check available models for your API key:
   ```bash
   python check_available_models.py
   ```
   This will show which Gemini models are available with your API key. If you get model errors, you can adjust the model name in `crew_ai_agents.py` (line 24).

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Select a dataset type:
   - **MNIST**: Pre-configured for handwritten digit recognition
   - **CIFAR-10**: Pre-configured for object recognition
   - **Custom**: Upload your own dataset summary

4. Click "Generate Neural Network Design" and wait for the AI agents to collaborate

5. Review the results:
   - Data Analysis
   - Architecture Design
   - Hyperparameter Recommendations
   - Ready-to-run PyTorch Training Script

## Project Structure

```
nn-hyperparameter-tuner/
├── app.py                 # Flask application
├── crew_ai_agents.py      # CrewAI agents and crew configuration
├── check_setup.py         # Setup verification script
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Uploaded dataset files (auto-created)
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment variables file
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## How It Works

The application uses CrewAI's sequential process flow:

1. **Data Analyst** receives the dataset information and analyzes its characteristics
2. **Model Designer** uses the analysis to propose an optimal architecture
3. **Hyperparameter Agent** suggests hyperparameters based on the dataset and architecture
4. **CodeGen Agent** generates a complete PyTorch script implementing everything

Each agent builds upon the previous agent's output, creating a collaborative design process.

## Customization

You can modify the agents' prompts and behavior in `crew_ai_agents.py`. The agents use `gemini-flash-latest` by default.

To use a different model, change the `model` parameter in the `LLM()` initialization in `crew_ai_agents.py` (line 24). Available options depend on your Google API access level.

**Note:** If you encounter quota errors (429), wait a few minutes and try again. The free tier has rate limits.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

