# SQL Evaluation Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive Python library for evaluating SQL query generation models with support for multiple evaluation strategies, database backends, and integration with Langfuse for experiment tracking.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git (for submodule management)
- Docker & Docker Compose (for Langfuse integration)

### Installation

#### Option 1: Pip Install (Recommended for Users)

```bash
pip install sql-evaluation-library
```

#### Option 2: Development Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd sql_finetuning/sql_evaluation_library

# Run the setup script (handles everything automatically)
bash scripts/setup_project.sh
```

The setup script will:
- ✅ Initialize git submodules (including Langfuse v3.66.1)
- ✅ Install UV package manager if needed
- ✅ Install project dependencies
- ✅ Set up environment configuration
- ✅ Validate package installation

### Quick Example

```python
from sql_eval_lib import SQLLLMEvaluator, SQLExecutionEvaluator, EvaluationSuite
from sql_eval_lib.dataset.models import DatasetItem

# Create a sample dataset item
item = DatasetItem(
    input={
        "question": "How many users are there?",
        "database_schema": "CREATE TABLE users (id INT, name TEXT);"
    },
    expected_output={"sql": "SELECT COUNT(*) FROM users;"},
    metadata={"difficulty": "basic"}
)

# Set up evaluation
evaluator = SQLLLMEvaluator(api_key="your-openai-key")
result = evaluator.evaluate_item(item)

print(f"Evaluation Score: {result.overall_score}")
```

## 📁 Project Structure

```
sql_evaluation_library/
├── src/sql_eval_lib/          # Main library code
│   ├── evaluation/            # Evaluation framework
│   │   ├── base.py           # Base classes
│   │   ├── sql/              # SQL-specific evaluators
│   │   └── metrics/          # Evaluation metrics
│   ├── dataset/              # Dataset management
│   │   ├── loaders/          # Data loaders (HuggingFace, etc.)
│   │   └── models.py         # Data models
│   ├── models/               # Model adapters
│   │   ├── adapters.py       # OpenAI, Anthropic, etc.
│   │   └── interface.py      # Model interface
│   └── langfuse/             # Langfuse integration
├── tests/                    # Test suite
│   ├── dataset/              # Dataset tests
│   ├── evaluation/           # Evaluation tests
│   └── conftest.py           # Test configuration
├── examples/                 # Example notebooks and scripts
│   ├── run_evaluation_notebook_refactored.ipynb
│   └── huggingface_eda.ipynb
├── scripts/                  # Utility scripts
│   └── setup_project.sh      # Project setup script
├── langfuse/                 # Langfuse submodule (v3.66.1)
└── pyproject.toml            # Project configuration
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=sql_eval_lib

# Run specific test categories
uv run pytest tests/dataset/
uv run pytest tests/evaluation/

# Run integration tests (requires Docker)
uv run pytest tests/ -m integration
```

### Integration Tests with Langfuse

The project includes end-to-end integration tests that automatically:

1. **Start Langfuse Docker Container**: Tests automatically spin up Langfuse using the submodule's docker-compose
2. **Run Evaluation Pipeline**: Execute complete evaluation workflows
3. **Verify Langfuse Logging**: Ensure traces and scores are properly logged
4. **Clean Up**: Automatically tear down test infrastructure

```bash
# Run integration tests (Docker required)
uv run pytest tests/langfuse/ -v

# Run integration tests with Langfuse logging
LANGFUSE_HOST=http://localhost:3000 uv run pytest tests/ -m langfuse
```

### Test Configuration

Tests are configured in `tests/conftest.py` with:
- Automatic Langfuse submodule initialization
- Shared test fixtures for datasets and evaluators
- Docker container management for integration tests
- Environment variable handling

## 📚 Examples & Documentation

### Jupyter Notebooks

The `examples/` directory contains comprehensive notebooks:

#### 1. **Evaluation Notebook** (`run_evaluation_notebook_refactored.ipynb`)
- Complete evaluation pipeline demonstration
- HuggingFace dataset integration
- Complex SQL query analysis
- Both standalone and Langfuse-integrated workflows

#### 2. **EDA Notebook** (`huggingface_eda.ipynb`)
- Exploratory data analysis of SQL datasets
- Dataset statistics and distribution analysis
- Complex JOIN query identification
- Training strategy recommendations

### Running Examples

```bash
# Install Jupyter if not already available
uv add jupyter

# Start Jupyter and open examples
uv run jupyter notebook examples/

# Or run specific notebook
uv run jupyter nbconvert --execute examples/run_evaluation_notebook_refactored.ipynb
```

## 🐳 Langfuse Integration

### Starting Langfuse

```bash
# Navigate to langfuse submodule and start services
cd langfuse
docker-compose up -d

# Check if services are running
docker-compose ps

# View logs
docker-compose logs -f
```

### Configuration

Set up your environment variables:

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your configuration
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=your_public_key
export LANGFUSE_SECRET_KEY=your_secret_key
```

### Using Langfuse in Code

```python
from sql_eval_lib.langfuse import LangfuseClient
from sql_eval_lib.dataset.langfuse_integration import LangfuseDatasetManager

# Initialize Langfuse client
client = LangfuseClient(
    host="http://localhost:3000",
    public_key="your_key",
    secret_key="your_secret"
)

# Create and manage datasets
dataset_manager = LangfuseDatasetManager(client)
dataset_id = dataset_manager.create_dataset(
    name="sql_evaluation_test",
    description="Test dataset for SQL evaluation"
)
```

## 🔧 Configuration

### Environment Variables

The library supports configuration via environment variables:

```bash
# Model Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_AI_API_KEY=your_google_key

# Langfuse Configuration
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key

# Database Configuration
DATABASE_TYPE=sqlite
DATABASE_CONNECTION_STRING=:memory:

# Evaluation Settings
EVALUATION_TIMEOUT=300
PARALLEL_EVALUATION=false
```

### Programmatic Configuration

```python
from sql_eval_lib.config import get_config

# Load configuration
config = get_config()

# Access configuration sections
print(f"Database backend: {config.database.backend_type}")
print(f"Langfuse enabled: {config.langfuse.enabled}")
print(f"Model adapter: {config.model.adapter_type}")
```

## 🏗️ Development

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Development Setup

```bash
# Clone with submodules
git clone --recursive <repo-url>
cd sql_finetuning/sql_evaluation_library

# Install in development mode
uv pip install -e .

# Install development dependencies
uv sync --dev

# Set up pre-commit hooks
pre-commit install
```

### Code Quality

The project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing

```bash
# Format code
uv run black src/ tests/

# Run linting
uv run flake8 src/ tests/

# Type checking
uv run mypy src/
```

## 📦 Building and Distribution

### Building the Package

```bash
# Build distribution packages
uv build

# Install locally for testing
uv pip install dist/sql_evaluation_library-*.whl
```

### Testing Package Installation

```bash
# Test installation in clean environment
python -m venv test_env
source test_env/bin/activate
pip install dist/sql_evaluation_library-*.whl

# Test import
python -c "import sql_eval_lib; print('✅ Package installed successfully')"
```

## 📊 Supported Features

### Evaluation Strategies
- **LLM-based Evaluation**: Using OpenAI, Anthropic, Google AI APIs
- **Execution-based Evaluation**: Direct SQL execution and result comparison
- **Syntax Validation**: SQL parsing and syntax checking
- **Semantic Similarity**: Vector-based similarity analysis

### Database Backends
- **SQLite**: In-memory and persistent databases
- **Trino**: Distributed SQL query engine
- **PostgreSQL**: Production database support
- **MySQL**: Popular relational database

### Model Adapters
- **OpenAI**: GPT-3.5, GPT-4 models
- **Anthropic**: Claude models
- **Google AI**: Gemini models
- **Ollama**: Self-hosted models
- **Custom**: Extensible interface for custom models

### Dataset Support
- **HuggingFace Datasets**: Direct integration with Hugging Face Hub
- **Custom Datasets**: Support for custom data formats
- **Langfuse Datasets**: Integration with Langfuse dataset management

## 🔗 Related Projects

- [Langfuse](https://github.com/langfuse/langfuse) - LLM Engineering Platform (v3.66.1)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/) - Dataset library
- [SQLGlot](https://github.com/tobymao/sqlglot) - SQL parser and transpiler

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- **Documentation**: [Coming Soon]
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: [Your Contact Email]

## 🎯 Roadmap

- [ ] Additional database backend support
- [ ] More model adapter integrations
- [ ] Advanced evaluation metrics
- [ ] Web-based evaluation dashboard
- [ ] Automated benchmarking workflows
- [ ] Integration with more ML platforms

---

**Happy SQL Evaluating! 🚀**