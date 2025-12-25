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

### Using this package in your project

Install `result-evaluator` as a dependency in your project:

#### With uv (recommended)

```bash
# From GitHub
uv add git+https://github.com/okhlynov/result-evaluator.git

# From local path (for development)
uv add --editable /path/to/result-evaluator

# From PyPI (once published)
uv add result-evaluator
```

#### With Poetry

```bash
# From GitHub
poetry add git+https://github.com/okhlynov/result-evaluator.git

# From local path (for development)
poetry add --editable /path/to/result-evaluator

# From PyPI (once published)
poetry add result-evaluator
```

#### With pip

```bash
# From GitHub
pip install git+https://github.com/okhlynov/result-evaluator.git

# From local path (for development)
pip install -e /path/to/result-evaluator

# From PyPI (once published)
pip install result-evaluator
```

### Development setup

If you want to contribute or modify this project:

#### Installing uv

If you don't have uv installed:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Clone and setup

```bash
# Clone the repository
git clone https://github.com/okhlynov/result-evaluator.git
cd result-evaluator

# Install dependencies (creates .venv automatically)
uv sync

# Run the application
uv run result-evaluator
```

## Configuration

The framework uses environment variables to configure the LLM client (OpenAI-compatible).
These variables use the default prefix `JUDGE_LLM_` but can be customized in code.

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `JUDGE_LLM_API_KEY` | OpenAI API key or LLM service API key | Yes | - |
| `JUDGE_LLM_MODEL` | Model name (e.g., `gpt-4o`) | Yes | - |
| `JUDGE_LLM_ENDPOINT` | Custom API endpoint URL (e.g., local Ollama) | No | `None` (uses OpenAI default) |
| `JUDGE_LLM_TIMEOUT` | Request timeout in seconds | No | `60` |
| `JUDGE_LLM_MAX_TOKENS` | Maximum tokens to generate | No | `None` (model default) |

**Note on Timeout:**
For complex reasoning or long outputs, increase `JUDGE_LLM_TIMEOUT`. The default is 60 seconds.

### LLM Configuration Examples

#### Using OpenAI (Default)

```bash
export JUDGE_LLM_API_KEY=sk-...
export JUDGE_LLM_MODEL=gpt-4o
export JUDGE_LLM_TIMEOUT=60
```

#### Using Local Ollama

For local semantic validation using Ollama, configure these environment variables:

```bash
export JUDGE_LLM_BASE_URL=http://localhost:11434/v1
export JUDGE_LLM_MODEL=llama3.2:latest
export JUDGE_LLM_API_KEY=ollama  # Ollama doesn't require a real API key, but some value is needed
export JUDGE_LLM_TIMEOUT=60  # Adjust for slower models
```

## Usage

### Sample YAML Test Case

Create a test scenario in YAML format:

```yaml
case:
  id: example_test
  description: Example test scenario
  tags: [api, validation]
input:
  user_id: 12345
  query: "test data"
run:
  kind: python
  target: tests.fixtures.dummy_inference
  timeout_ms: 5000
asserts:
  - path: $.status
    op: equals
    expected: success
  - path: $.count
    op: equals
    expected: 42
```

### Running Scenarios

```python
from result_evaluator import Engine, load_test_case

# Load test case from YAML file
test_case = load_test_case("test_case.yaml")

# Run the test
engine = Engine()
result = engine.run_test(test_case)

# Check results
print(f"Status: {result['status']}")  # PASS, FAIL, or ERROR
print(f"Case ID: {result['case_id']}")
for assert_result in result['asserts']:
    status = "✓" if assert_result['ok'] else "✗"
    print(f"  {status} Assert {assert_result['index']}: {assert_result['message']}")
```

Or with manual YAML parsing:

```python
import yaml
from result_evaluator import Scenario, Engine

# Load test case from YAML
with open("test_case.yaml") as f:
    yaml_data = yaml.safe_load(f)
    test_case = Scenario.model_validate(yaml_data)

# Run the test
engine = Engine()
result = engine.run_test(test_case)
```

### Operator-Specific Configuration

Some operators support additional configuration through the `config` field. This allows you to customize operator behavior without modifying the core framework.

**Configuration Validation:**
- The `config` field is optional and defaults to `None`
- Configuration validation happens at runtime when the operator is executed
- Invalid or missing required config keys will result in assertion failure

**Example with config:**
```yaml
asserts:
  - op: llm_judge
    path: $.answer
    expected: "true"
    config:
      prompt: "Is {input} semantically equivalent to {expected}?"
      system_prompt: "You are a fair judge. Respond only with true or false."
      response_path: "$.verdict"
```

### Available Operators

#### `exists` - Check if value exists and is not empty

```yaml
asserts:
  - path: $.user.name
    op: exists
```

#### `equals` - Strict equality check

```yaml
asserts:
  - path: $.status
    op: equals
    expected: success
  - path: $.count
    op: equals
    expected: 42
```

#### `contains` - Check if element is in string or list

```yaml
asserts:
  # String contains
  - path: $.message
    op: contains
    expected: "error"
  # List contains
  - path: $.tags
    op: contains
    expected: "production"
```

#### `not_contains` - Check if element is not in string or list

```yaml
asserts:
  # String contains
  - path: $.message
    op: not_contains
    expected: "error"
  # List contains
  - path: $.tags
    op: not_contains
    expected: "production"
```

#### `length_ge` - Check if length is greater than or equal

```yaml
asserts:
  - path: $.results
    op: length_ge
    expected: 10
  - path: $.name
    op: length_ge
    expected: 5
```

#### `match_regex` - Match against regular expression

```yaml
asserts:
  - path: $.email
    op: match_regex
    expected: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
  - path: $.version
    op: match_regex
    expected: "^v\\d+\\.\\d+\\.\\d+$"
```

#### `sequence_in_order` - Check ordered sequence in string list

Validates that expected string items appear in order within the top N items of a list. Additional items can appear between expected items, but the expected items must maintain their specified order.

**Parameters:**
- `expected` (object): Object containing:
  - `data` (list[str]): List of strings that must appear in order
  - `limit` (int): Number of top items from the selection to evaluate

**Use case:** Validate event logs, sequential data, or process steps where order matters but noise is expected.

```yaml
asserts:
  # Check that events occur in expected order within first 10 items
  - path: $.events[*].type
    op: sequence_in_order
    expected:
      data:
        - "START"
        - "PROCESSING"
        - "COMPLETE"
      limit: 10

  # Validate log messages appear in correct sequence
  - path: $.logs[*].level
    op: sequence_in_order
    expected:
      data:
        - "INFO"
        - "WARNING"
        - "ERROR"
      limit: 20
```

#### `llm_judge` - Semantic equivalence validation using LLM

Validates semantic equivalence between actual output and ground truth using an LLM. Unlike strict comparison operators (`equals`, `contains`), `llm_judge` can recognize semantically identical outputs that differ structurally or lexically.

**Use case:** Validate that outputs are semantically correct even when they use different wording, structure, or phrasing.

**Parameters:**
- `ground_truth` (string | dict): Required. The expected value to compare against.
- `expected` (bool): Optional, defaults to `true`. Set to `false` to test for semantic non-equivalence.
- `prompt` (string): Optional. Custom user prompt template with `{actual}` and `{ground_truth}` placeholders.
- `system_prompt` (string): Optional. Custom system prompt for the LLM.

**Cost Considerations:** Each assertion using `llm_judge` makes an LLM API call. For OpenAI, this incurs API costs. Consider using `llm_judge` selectively for scenarios where semantic validation is necessary.

**Example 1: Basic semantic string comparison**

```yaml
asserts:
  - path: $.answer
    op: llm_judge
    config:
      ground_truth: "Paris is the capital of France"
```

This will pass for outputs like:
- "The capital city of France is Paris"
- "Paris"
- "France's capital is Paris"

**Example 2: Testing for non-equivalence**

```yaml
asserts:
  - path: $.answer
    op: llm_judge
    config:
      ground_truth: "Paris is the capital of France"
      expected: false  # Expecting semantic difference
```

This assertion passes only if the LLM determines the actual output is NOT semantically equivalent to the ground truth.

**Example 3: Custom prompt template**

```yaml
asserts:
  - path: $.summary
    op: llm_judge
    config:
      ground_truth: "The user wants to build a search engine for documentation"
      prompt: |
        Does the actual summary accurately capture the essence of the ground truth requirement?

        Actual summary: {actual}
        Ground truth: {ground_truth}

        Consider completeness, accuracy, and key details.
```

**Example 4: Comparing complex data structures**

```yaml
asserts:
  - path: $.entities
    op: llm_judge
    config:
      ground_truth:
        person: "Marie Curie"
        field: "Physics and Chemistry"
        achievement: "Nobel Prize winner"
      prompt: |
        Compare the extracted entities with the ground truth.

        Extracted: {actual}
        Expected: {ground_truth}

        Check if the key information is semantically equivalent, even if structure differs.
```

## Tutorial

New to result-evaluator or the `llm_judge` operator? Start with our interactive tutorial:

```bash
# Quick start (2 commands)
source tutorial/ollama.env
./tutorial/run_tutorial.sh

# Run specific tests
./tutorial/run_tutorial.sh "01-*.yaml"

# Or use the runner directly
PYTHONPATH=. uv run python tutorial/run_evaluator.py --dataset "01-*.yaml"
```

The tutorial demonstrates semantic validation using production-style YAML test cases and structured logging. You'll learn:
- When to use `llm_judge` vs traditional operators
- How to write declarative YAML test cases with the Engine
- How to configure local Ollama for LLM-based testing
- Production integration patterns with structured JSONL logging

See **[TUTORIAL.md](TUTORIAL.md)** for the comprehensive guide.

**Cost note**: The tutorial uses Ollama (free, runs locally). For production, you can switch to OpenAI API (paid).

### Assertion Composition

Combine assertions with logical operators:

#### `all` - All nested rules must pass (AND logic)

```yaml
asserts:
  - op: ""
    all:
      - path: $.status
        op: equals
        expected: success
      - path: $.count
        op: length_ge
        expected: 1
```

**With config:**
```yaml
asserts:
  - op: ""
    all:
      - path: $.answer
        op: llm_judge
        expected: "true"
        config:
          prompt: "Is the answer correct?"
          response_path: "$.verdict"
      - path: $.confidence
        op: length_ge
        expected: 0.7
```

#### `any` - At least one nested rule must pass (OR logic)

```yaml
asserts:
  - op: ""
    any:
      - path: $.status
        op: equals
        expected: success
      - path: $.status
        op: equals
        expected: completed
```

#### `not` - Invert the nested rule result

```yaml
asserts:
  - op: ""
    not:
      path: $.error
      op: exists
```

### CLI Usage

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

