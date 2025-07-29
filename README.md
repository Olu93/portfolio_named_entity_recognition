# Named Entity Recognition (NER) API

A comprehensive Named Entity Recognition system that extracts persons, organizations, and locations from text using various machine learning models including BERT, SpaCy, and OpenAI GPT models.

## 📋 Technical Documentation

**🔬 [TECHNICAL REPORT](https://drive.google.com/file/d/1dYhbSJpb2pVCUGEr-efEVt6YyT3nW5kk/view?usp=sharing)**

This comprehensive technical report contains:
- Detailed development procedures and methodology
- Model selection and evaluation criteria
- Performance analysis and benchmarking
- Technical decisions and architectural choices
- Experimental results and findings

**📖 [TASK DESCRIPTION](TASK_DESCRIPTION.md)**

The original task requirements and specifications.

## Project Overview

This project provides a FastAPI-based service for extracting named entities (persons, organizations, and locations) from text using multiple model configurations. The system supports various model types:

- **Transformer Models**: BERT-based models (HuggingFace, custom fine-tuned)
- **Rule-based Models**: SpaCy models (small, medium, large)
- **LLM Models**: OpenAI GPT-4o-mini via LangChain
- **Naive Models**: Sliding window approach

## Environment Variables

The following environment variables control the model configuration and training process:

### Model Configuration Variables
These variables determine which models are used for each entity type:

- `MODEL_NAME_PERSON`: The class name of the model for person extraction
- `MODEL_VARIANT_PERSON`: The specific model variant for person extraction
- `MODEL_NAME_ORGANISATION`: The class name of the model for organization extraction  
- `MODEL_VARIANT_ORGANISATION`: The specific model variant for organization extraction
- `MODEL_NAME_LOCATION`: The class name of the model for location extraction
- `MODEL_VARIANT_LOCATION`: The specific model variant for location extraction

### Training Data Variable
- `TRAIN_DATA_PATH`: Path to the training dataset file (relative to the project root)


## Model Configuration

The available models are defined in `src/constants/model_config.py`. Each model configuration includes:

- **Name**: The class name of the extractor
- **Extra Info**: Description, model variant, type, and paper reference
- **Entity-specific configurations**: Parameters for persons, organizations, and locations

### Available Models

1. **LangChainEntityExtractor** (gpt-4o-mini)
2. **PretrainedBERTEntityExtractor** (bert-base-cased-finetuned, dslim-bert-base-cased-finetuned)
3. **HuggingFaceEntityExtractor** (dslim/bert-base-NER, dslim/distilbert-NER, dbmdz/bert-large-cased-finetuned-conll03-english)
4. **SpacyEntityExtractor** (en_core_web_sm, en_core_web_md, en_core_web_lg)
5. **SlidingWindowExtractor** (naive rule-based approach)

## Installation

### Prerequisites

- Python 3.12
- Poetry (for dependency management)
- Docker and Docker Compose (for containerized deployment)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd take-home-assignment
   ```

2. **Install Poetry** (if not already installed)
   ```bash
   pip install poetry
   ```

3. **Install dependencies**
   ```bash
   poetry install
   ```

4. **Install SpaCy models** (required for SpaCy-based extractors)
   ```bash
   poetry run python -m spacy download en_core_web_sm
   poetry run python -m spacy download en_core_web_md
   poetry run python -m spacy download en_core_web_lg
   ```

5. **Set up environment variables**
   Create a `.env` file in the project root with your configuration:
   ```bash
   # Model Configuration
   MODEL_NAME_PERSON=HuggingFaceEntityExtractor
   MODEL_VARIANT_PERSON=dslim/bert-base-NER
   MODEL_NAME_ORGANISATION=HuggingFaceEntityExtractor
   MODEL_VARIANT_ORGANISATION=dslim/bert-base-NER
   MODEL_NAME_LOCATION=HuggingFaceEntityExtractor
   MODEL_VARIANT_LOCATION=dslim/bert-base-NER
   
   # Training Data
   TRAIN_DATA_PATH=files/datasets/full_data_clean.csv
   
   # Logging (optional)
   LOG_LEVEL=INFO
   LOG_FILE_PATH=logs/app.log
   LOG_MAX_BYTES=10485760
   LOG_BACKUP_COUNT=5
   ```

## Usage

### Step 1: Train and Save Models

Before running the API, you must train and save the models using the provided notebook:

```bash
# Run the training script
poetry run python notebooks/07_train_and_save.py
```

This script will:
- Load the specified models based on environment variables
- Train them on the provided dataset
- Save the trained models to the `models/` directory with a timestamp

### Step 2: Run the Application

#### Local Development

```bash
cd src
# Run the FastAPI application
poetry run python -m main
```

The API will be available at `http://localhost:8000`

**API Documentation**: Since this is a FastAPI application, interactive Swagger/OpenAPI documentation is automatically generated and available at `http://localhost:8000/docs`

#### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build
```

The API will be available at `http://localhost:8000`

**API Documentation**: Since this is a FastAPI application, interactive Swagger/OpenAPI documentation is automatically generated and available at `http://localhost:8000/docs`

## API Endpoints

- `GET /`: Root endpoint
- `GET /healthz`: Health check endpoint
- `POST /predict/single`: Extract entities from a single text
- `POST /predict/bulk`: Extract entities from a CSV file
- `GET /download/{file_path}`: Download processed files
- `GET /model-info`: Get information about the loaded model




## Quick Setup Script

For convenience, a setup script is provided that automates the training and deployment process:

```bash
# Make the script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

The setup script will:
1. Check if Poetry is installed
2. Install dependencies
3. Download SpaCy models
4. Train and save the models
5. Start the Docker container

## Project Structure

```
take-home-assignment/
├── src/                    # Source code
│   ├── adapter/           # Model adapters
│   ├── constants/         # Configuration constants
│   ├── port/              # Entity extraction interface
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for training
├── files/                 # Data files and models
├── models/                # Saved trained models
├── docker-compose.yml     # Docker configuration
├── Dockerfile            # Container definition
└── pyproject.toml        # Project dependencies
```

