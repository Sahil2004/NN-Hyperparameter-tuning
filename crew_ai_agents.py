from crewai import Agent, Task, Crew, Process, LLM
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize LLM - using Google Gemini ONLY
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "your-api-key-here":
    raise ValueError(
        "Please set your GOOGLE_API_KEY or GEMINI_API_KEY environment variable or create a .env file. "
        "You can get a free API key from https://aistudio.google.com/u/1/api-keys"
    )

# Set the API key for Google Gemini (CrewAI native provider expects GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize LLM with Google Gemini using CrewAI's native provider
# Using gemini-flash-latest as requested
llm = LLM(
    model="gemini-flash-latest",
    temperature=0.7,
    api_key=api_key
)


def create_data_analyst_agent():
    """Agent that analyzes dataset characteristics and constraints"""
    return Agent(
        role='Data Analyst',
        goal='Analyze the dataset to identify its characteristics, constraints, and requirements for neural network training',
        backstory="""You are an expert data analyst with deep experience in understanding 
        dataset properties. You excel at identifying data types, dimensions, class distributions, 
        and specific constraints that will guide neural network design decisions.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def create_model_designer_agent():
    """Agent that proposes neural network architecture"""
    return Agent(
        role='Neural Network Architect',
        goal='Design the optimal neural network architecture based on dataset characteristics',
        backstory="""You are a world-class neural network architect with expertise in designing 
        MLPs, CNNs, RNNs, and Transformers. You understand when to use each architecture type 
        and how to structure layers for optimal performance on specific tasks.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def create_hyperparameter_agent():
    """Agent that suggests optimal hyperparameters"""
    return Agent(
        role='Hyperparameter Optimization Specialist',
        goal='Recommend optimal hyperparameters including learning rate, optimizer, batch size, and training epochs',
        backstory="""You are a hyperparameter tuning expert with extensive knowledge of 
        optimization strategies. You understand the relationships between learning rates, 
        optimizers, batch sizes, and other hyperparameters, and can recommend optimal settings 
        based on dataset and architecture characteristics.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def create_codegen_agent():
    """Agent that generates PyTorch training code"""
    return Agent(
        role='PyTorch Code Generator',
        goal='Generate complete, ready-to-run PyTorch training scripts based on architecture and hyperparameter specifications',
        backstory="""You are an expert PyTorch developer who writes clean, efficient, 
        and well-documented code. You excel at implementing neural networks, data loading, 
        training loops, and evaluation metrics. Your code is production-ready and follows 
        best practices.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def create_crew():
    """Create and configure the CrewAI crew with all agents"""
    data_analyst = create_data_analyst_agent()
    model_designer = create_model_designer_agent()
    hyperparam_agent = create_hyperparameter_agent()
    codegen_agent = create_codegen_agent()
    
    return {
        'data_analyst': data_analyst,
        'model_designer': model_designer,
        'hyperparam_agent': hyperparam_agent,
        'codegen_agent': codegen_agent
    }


def process_dataset_request(context: Dict[str, Any]) -> Dict[str, Any]:
    """Process dataset request through all agents in sequence"""
    
    agents = create_crew()
    
    dataset_info = f"""
    Dataset Type: {context.get('dataset_type', 'unknown')}
    Dataset Summary: {context.get('dataset_summary', 'No summary provided')}
    """
    
    # Task 1: Data Analysis
    analysis_task = Task(
        description=f"""
        Analyze the following dataset information and provide detailed characteristics:
        
        {dataset_info}
        
        Provide a comprehensive analysis including:
        1. Data type (images, text, tabular, etc.)
        2. Input dimensions/shape
        3. Number of classes (if classification) or output type (if regression)
        4. Dataset size (training/validation/test splits)
        5. Any special characteristics or constraints
        6. Recommended preprocessing steps
        
        Format your output clearly with labeled sections. Start with "DATA ANALYSIS:" header.
        """,
        agent=agents['data_analyst'],
        expected_output="A comprehensive analysis of dataset characteristics with clear sections"
    )
    
    # Task 2: Architecture Design - depends on analysis_task
    design_task = Task(
        description=f"""
        Based on the data analysis from the previous agent, design the optimal neural network architecture.
        
        Dataset Information:
        {dataset_info}
        
        Review the data analysis output and consider:
        - Whether to use MLP, CNN, RNN, or other architectures
        - Number and types of layers
        - Activation functions
        - Regularization techniques (dropout, batch norm, etc.)
        - Architecture justification
        
        Provide:
        1. Architecture type and reasoning
        2. Layer-by-layer specification with dimensions
        3. Justification for architectural choices
        4. Expected input/output shapes
        
        Format your output clearly. Start with "ARCHITECTURE DESIGN:" header.
        """,
        agent=agents['model_designer'],
        expected_output="Detailed neural network architecture specification with justification",
        context=[analysis_task]  # This task can see the output of analysis_task
    )
    
    # Task 3: Hyperparameter Recommendation - depends on previous tasks
    hyperparam_task = Task(
        description=f"""
        Based on the dataset analysis and proposed architecture from previous agents, recommend optimal hyperparameters.
        
        Dataset Information:
        {dataset_info}
        
        Review all previous outputs and specify:
        1. Learning rate (with justification)
        2. Optimizer (SGD, Adam, AdamW, etc.) and its parameters
        3. Batch size
        4. Number of training epochs
        5. Loss function
        6. Evaluation metrics
        7. Any learning rate scheduling
        8. Regularization parameters (if applicable)
        
        Justify each choice based on the dataset and architecture characteristics.
        
        Format your output clearly. Start with "HYPERPARAMETERS:" header.
        """,
        agent=agents['hyperparam_agent'],
        expected_output="Comprehensive hyperparameter recommendations with justifications",
        context=[analysis_task, design_task]  # Can see outputs of previous tasks
    )
    
    # Task 4: Code Generation - depends on all previous tasks
    codegen_task = Task(
        description=f"""
        Generate a complete, ready-to-run PyTorch training script based on all previous agents' outputs.
        
        Dataset Information:
        {dataset_info}
        
        Review the data analysis, architecture design, and hyperparameter recommendations.
        
        The script must:
        1. Implement the proposed architecture exactly as specified
        2. Use the recommended hyperparameters
        3. Include data loading (support MNIST and CIFAR-10 from torchvision, or custom dataset loading)
        4. Implement training loop with proper logging
        5. Include validation/testing
        6. Save/load model checkpoints
        7. Have clear comments and documentation
        8. Be production-ready and executable
        
        The code should be complete and executable. Include all necessary imports and helper functions.
        Wrap your code output in ```python code blocks.
        
        Start with "TRAINING SCRIPT:" header.
        """,
        agent=agents['codegen_agent'],
        expected_output="Complete PyTorch training script ready to run, wrapped in code blocks",
        context=[analysis_task, design_task, hyperparam_task]  # Can see all previous outputs
    )
    
    # Create crew with sequential execution
    # Configure LLM at crew level for better compatibility
    crew = Crew(
        agents=[
            agents['data_analyst'],
            agents['model_designer'],
            agents['hyperparam_agent'],
            agents['codegen_agent']
        ],
        tasks=[
            analysis_task,
            design_task,
            hyperparam_task,
            codegen_task
        ],
        process=Process.sequential,  # Run tasks sequentially so each can use previous output
        verbose=True
    )
    
    # Execute the crew
    result = crew.kickoff(inputs=context)
    
    # Extract results from task outputs
    full_output = str(result)
    
    # Try to extract structured sections from the output
    data_analysis = extract_section_by_header(full_output, "DATA ANALYSIS:")
    architecture = extract_section_by_header(full_output, "ARCHITECTURE DESIGN:")
    hyperparameters = extract_section_by_header(full_output, "HYPERPARAMETERS:")
    training_script = extract_code_block(full_output)
    
    # If extraction failed, try to get from individual task outputs
    if not data_analysis and hasattr(result, 'tasks_output'):
        try:
            data_analysis = str(result.tasks_output[0].raw) if len(result.tasks_output) > 0 else full_output
        except:
            pass
    
    if not architecture and hasattr(result, 'tasks_output'):
        try:
            architecture = str(result.tasks_output[1].raw) if len(result.tasks_output) > 1 else ""
        except:
            pass
    
    if not hyperparameters and hasattr(result, 'tasks_output'):
        try:
            hyperparameters = str(result.tasks_output[2].raw) if len(result.tasks_output) > 2 else ""
        except:
            pass
    
    if not training_script and hasattr(result, 'tasks_output'):
        try:
            training_script = str(result.tasks_output[3].raw) if len(result.tasks_output) > 3 else ""
        except:
            pass
    
    return {
        'data_analysis': data_analysis or full_output,
        'architecture': architecture or full_output,
        'hyperparameters': hyperparameters or full_output,
        'training_script': training_script or full_output,
        'full_output': full_output
    }


def extract_section_by_header(text: str, header: str) -> str:
    """Extract section content by header"""
    if not isinstance(text, str):
        text = str(text)
    
    header_index = text.find(header)
    if header_index == -1:
        return ""
    
    # Find the start of content after header
    content_start = header_index + len(header)
    
    # Find the next header or end of text
    next_headers = [
        "ARCHITECTURE DESIGN:",
        "HYPERPARAMETERS:",
        "TRAINING SCRIPT:",
        "DATA ANALYSIS:"
    ]
    
    next_header_pos = len(text)
    for h in next_headers:
        if h != header:
            pos = text.find(h, content_start)
            if pos != -1 and pos < next_header_pos:
                next_header_pos = pos
    
    section_content = text[content_start:next_header_pos].strip()
    return section_content


def extract_code_block(text: str) -> str:
    """Extract Python code block from text"""
    if not isinstance(text, str):
        text = str(text)
    
    # Look for code blocks
    import re
    pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Return the largest code block (likely the training script)
        return max(matches, key=len).strip()
    
    # If no code block found, look for "TRAINING SCRIPT:" section
    return extract_section_by_header(text, "TRAINING SCRIPT:")


def extract_section(raw_output: str, section: str) -> str:
    """Extract specific sections from crew output (legacy function)"""
    if isinstance(raw_output, str):
        return raw_output
    return str(raw_output)

