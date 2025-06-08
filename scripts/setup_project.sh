#!/bin/bash

# LLMWatch - Project Setup Script
# This script sets up the project environment including submodules and dependencies

set -e  # Exit on any error

echo "🚀 Setting up LLMWatch Project"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}📋 $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Error: pyproject.toml not found. Please run this script from the llmwatch directory."
    exit 1
fi

print_info "Step 1: Initializing Git Submodules"
if [ -d ".git" ]; then
    git submodule update --init --recursive
    print_status "Git submodules initialized"
else
    print_warning "Not a git repository - skipping submodule initialization"
fi

print_info "Step 2: Checking Python and UV"
# Check if Python 3.10+ is available
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        print_status "Python $PYTHON_VERSION found (3.10+ required)"
    else
        print_error "Python 3.10+ is required but found $PYTHON_VERSION. Please install Python 3.10 or higher."
        exit 1
    fi
else
    print_error "Python 3 is required but not found. Please install Python 3.10 or higher."
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_warning "UV package manager not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    print_status "UV installed successfully"
else
    print_status "UV package manager found"
fi

print_info "Step 3: Installing Project Dependencies"
uv sync
print_status "Dependencies installed"

print_info "Step 4: Setting up Environment Configuration"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_status "Created .env file from template"
        print_warning "Please edit .env file with your configuration"
    else
        print_warning "No .env.example found - you may need to create .env manually"
    fi
else
    print_status ".env file already exists"
fi

print_info "Step 5: Running Quick Validation"
# Test that the package can be imported
if uv run python -c "import sql_eval_lib; print('✅ Package import successful')" 2>/dev/null; then
    print_status "Package validation successful"
else
    print_warning "Package import validation failed - this may be expected if dependencies are missing"
fi

print_info "Step 6: Langfuse Setup Check"
if [ -d "langfuse" ] && [ -f "langfuse/docker-compose.yml" ]; then
    print_status "Langfuse submodule ready"
    print_info "To start Langfuse for testing:"
    echo "  cd langfuse && docker-compose up -d"
else
    print_warning "Langfuse submodule not found or incomplete"
fi

echo ""
echo "🎉 Project Setup Complete!"
echo "=========================="
print_info "Next steps:"
echo "  1. Edit .env file with your API keys and configuration"
echo "  2. Start Langfuse if needed: cd langfuse && docker-compose up -d"
echo "  3. Run tests: uv run pytest"
echo "  4. Check examples: see examples/ directory"
echo "  5. Install in development mode: uv pip install -e ."
echo ""
print_status "Happy coding! 🚀" 