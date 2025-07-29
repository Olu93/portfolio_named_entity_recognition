#!/bin/bash

# Setup script for the NER API project
# This script automates the training and deployment process

set -e  # Exit on any error

echo "🚀 Starting NER API setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Poetry is installed
check_poetry() {
    print_status "Checking if Poetry is installed..."
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry is not installed. Please install Poetry first:"
        echo "pip install poetry"
        exit 1
    fi
    print_success "Poetry is installed"
}

# Install dependencies
install_dependencies() {
    print_status "Installing project dependencies..."
    poetry install
    print_success "Dependencies installed successfully"
}

# Install SpaCy models
install_spacy_models() {
    print_status "Installing SpaCy models..."
    poetry run python -m spacy download en_core_web_sm
    poetry run python -m spacy download en_core_web_md
    poetry run python -m spacy download en_core_web_lg
    print_success "SpaCy models installed successfully"
}

# Check if .env file exists
check_env_file() {
    print_status "Checking for .env file..."
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating a default .env file..."
        cat > .env << EOF
# Model Configuration
MODEL_NAME_PERSON=HuggingFaceEntityExtractor
MODEL_VARIANT_PERSON=dslim/bert-base-NER
MODEL_NAME_ORGANISATION=HuggingFaceEntityExtractor
MODEL_VARIANT_ORGANISATION=dslim/bert-base-NER
MODEL_NAME_LOCATION=HuggingFaceEntityExtractor
MODEL_VARIANT_LOCATION=dslim/bert-base-NER

# Training Data
TRAIN_DATA_PATH=files/datasets/full_data_clean_finetune.csv

# Logging (optional)
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/app.log
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
EOF
        print_success "Default .env file created. Please review and modify as needed."
    else
        print_success ".env file found"
    fi
}

# Train and save models
train_models() {
    print_status "Training and saving models..."
    
    # Check if training data exists
    if [ ! -f "files/datasets/full_data_clean_finetune.csv" ]; then
        print_error "Training data not found at files/datasets/full_data_clean_finetune.csv"
        print_error "Please ensure the training dataset is available before running this script"
        exit 1
    fi
    
    # Run the training script
    poetry run python notebooks/07_train_and_save.py
    
    if [ $? -eq 0 ]; then
        print_success "Models trained and saved successfully"
    else
        print_error "Model training failed"
        exit 1
    fi
}

# Check if Docker is available
check_docker() {
    print_status "Checking if Docker is available..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are available"
}

# Start Docker container
start_docker() {
    print_status "Starting Docker container..."
    docker-compose up --build -d
    
    if [ $? -eq 0 ]; then
        print_success "Docker container started successfully"
        echo ""
        echo "🎉 Setup completed successfully!"
        echo "📊 API is now running at: http://localhost:8000"
        echo "📖 API documentation: http://localhost:8000/docs"
        echo "🏥 Health check: http://localhost:8000/healthz"
        echo ""
        echo "To view logs: docker-compose logs -f"
        echo "To stop the service: docker-compose down"
    else
        print_error "Failed to start Docker container"
        exit 1
    fi
}

# Main execution
main() {
    echo "=========================================="
    echo "    NER API Setup Script"
    echo "=========================================="
    echo ""
    
    check_poetry
    install_dependencies
    install_spacy_models
    check_env_file
    train_models
    check_docker
    start_docker
}

# Run main function
main "$@" 