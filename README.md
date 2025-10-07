# Result Evaluator

Test framework to evaluate results with a complex structure against groundtruth dataset

## Features

- Built with modern Python 3.12+ features
- Type-safe with comprehensive type hints
- Fully tested with pytest
- Linted and formatted with Ruff
- Type-checked with Pyright

## Requirements

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

### Installing uv

If you don't have uv installed:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installing the project

```bash
# Clone the repository
git clone <repository-url>
cd result-evaluator

# Install dependencies (creates .venv automatically)
uv sync

# Run the application
uv run result-evaluator
```

## Usage

```bash
# Run the CLI tool
uv run result-evaluator

# Or activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# .venv\Scripts\activate   # On Windows
result-evaluator
```

## Development

### Setup Development Environment

```bash
# Install all dependencies including dev dependencies
uv sync

# The dev dependencies include:
# - pytest: Testing framework
# - ruff: Linting and formatting
# - pyright: Type checking
```

### Code Quality Tools

```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check

# Lint and auto-fix issues
uv run ruff check --fix

# Type check
uv run pyright

# Run all checks
uv run ruff format && uv run ruff check && uv run pyright
```

### Testing

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=result_evaluator

# Run specific test file
uv run pytest tests/test_main.py

# Run tests in verbose mode
uv run pytest -v
```

### Adding Dependencies

```bash
# Add a runtime dependency
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Remove a dependency
uv remove <package-name>
```

## Project Structure

```
result-evaluator/
├── src/
│   └── result_evaluator/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── pyproject.toml
├── pyrightconfig.json
├── .python-version
├── .gitignore
└── README.md
```

## Configuration

### Ruff Configuration

The project uses Ruff for both linting and formatting with the following rules:

- **E4/E7/E9**: pycodestyle errors
- **F**: Pyflakes (basic errors)
- **B**: flake8-bugbear (common bugs)
- **I**: isort (import sorting)
- **UP**: pyupgrade (modernize to Python 3.12 syntax)
- **ANN**: flake8-annotations (require type hints)
- **FA**: flake8-future-annotations

Configuration is in `pyproject.toml` under `[tool.ruff]`.

### Pyright Configuration

Type checking is configured in `pyrightconfig.json` with:

- Type checking mode: standard
- Python version: 3.12
- Included: src/
- Excluded: caches, venv, build artifacts

## Python 3.12 Features

This project leverages modern Python 3.12 features:

- PEP 695 type parameter syntax
- Modern type hints
- Type statement for type aliases

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run all quality checks (format, lint, type check, test)
5. Submit a pull request

## License

[Choose your license]

## Authors

- Hlinov Oleg <o.hlinov@cmit.mipt.ru>
