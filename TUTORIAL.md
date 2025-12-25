# Tutorial: Semantic Validation with llm_judge

> Learn how to use the `llm_judge` operator for semantic matching in your tests

## What You'll Learn

- How to use the `llm_judge` operator for semantic validation
- When to use semantic matching vs strict comparison operators
- How to configure Ollama for local LLM-based testing
- Practical patterns for production integration

## Why llm_judge?

### The Problem with Strict Matching

Traditional assertion operators use exact matching:

```python
# With strict 'equals' operator
assert category == "Electronics"  # Fails for "Electronics & Technology"

# With 'contains' operator
assert "Electronics" in category  # Fails for "Electronic Devices"
```

This creates brittle tests that break when:
- Wording changes slightly ("Electronics" → "Electronic Products")
- Format changes ("Electronics & Tech" → "Electronics and Technology")
- Synonyms are used ("Books" vs "Literature")

### The llm_judge Solution

The `llm_judge` operator uses an LLM to understand semantic equivalence:

```python
# With llm_judge
result = op_llm_judge(
    "Electronics & Technology",
    {"ground_truth": "Electronics"}
)
# ✓ Passes - semantically equivalent
```

**When to use llm_judge:**
- Validating natural language outputs
- Checking if concepts are present (not exact wording)
- Testing against user-generated content
- Verifying semantic correctness in AI outputs

**When NOT to use llm_judge:**
- Exact string matching needed (use `equals`)
- Performance-critical paths (LLM calls are slower)
- Simple substring checks (use `contains`)
- Structured data comparison (use `equals` with dicts)

## Prerequisites

Before starting this tutorial, ensure you have:

### 1. Ollama Installed and Running

```bash
# Install Ollama
# Visit: https://ollama.ai

# Verify installation
ollama --version

# Start Ollama server
ollama serve
```

### 2. Model Downloaded

```bash
# Pull the recommended model
ollama pull llama3.2

# Verify model is available
ollama list
```

### 3. Project Dependencies

```bash
# Ensure project dependencies are installed
uv sync

# Verify pytest is available
uv run pytest --version
```

## Quick Start

Get the tutorial running in 2 steps:

```bash
# Step 1: Configure environment
source tutorial/ollama.env

# Step 2: Run tutorial tests
./tutorial/run_tutorial.sh

# Run specific test cases
./tutorial/run_tutorial.sh "01-*.yaml"
./tutorial/run_tutorial.sh "dataset/02-exact-match.yaml"

# Alternative: Run evaluator directly
PYTHONPATH=. uv run python tutorial/run_evaluator.py
PYTHONPATH=. uv run python tutorial/run_evaluator.py --dataset "01-*.yaml"
```

### Viewing LLM Prompts (Debug Mode)

The runner automatically logs all test execution to `tutorial/evaluation_log.jsonl` in structured JSONL format, including the exact LLM prompts at DEBUG level.

**View all LLM prompts:**
```bash
# Extract and display all LLM prompts
grep "LLM Call" tutorial/evaluation_log.jsonl | jq -r '.message'

# View full log entries with timestamps
grep "LLM Call" tutorial/evaluation_log.jsonl | jq '.'

# Pretty-print the entire log file
cat tutorial/evaluation_log.jsonl | jq '.'
```

**View prompts for specific test:**
```bash
# Run a single test to see its prompts
./tutorial/run_tutorial.sh "01-semantic-match.yaml"
cat tutorial/evaluation_log.jsonl | grep "LLM Call"
```

The logs capture:
- **System prompt**: LLM instructions and behavioral guidelines
- **User prompt**: Actual data and comparison ground truth
- **Test results**: Pass/fail status and assertion details
- **Timestamps**: When each operation occurred
- **All DEBUG messages**: Including LLM call metadata

Expected output:
```
==========================================
 llm_judge Tutorial Runner
==========================================

➜ Checking Ollama availability...
✓ Ollama is running and accessible

➜ Loading environment configuration...
✓ Environment loaded from ollama.env

...

==========================================
 Tutorial Completed Successfully
==========================================
✓ All tests passed!
```

## How It Works

### The Function Under Test

Our tutorial uses a product categorization function:

```python
def get_product_categories() -> list[str]:
    """Return list of product categories."""
    return [
        "Electronics & Technology",
        "Home & Kitchen Appliances",
        "Fashion & Clothing",
        # ... more categories
    ]
```

This function returns categories with semantic variations:
- "Electronics & Technology" (not just "Electronics")
- "Books, Media & Entertainment" (not just "Books")
- "Health & Beauty Products" (not just "Beauty")

Perfect for demonstrating why semantic matching is valuable!

### Understanding YAML Test Cases

This tutorial uses **production-style YAML test cases** with the `result_evaluator` Engine. This approach mirrors how you'd use result-evaluator in real applications.

#### YAML Test Case Structure

Each test case is defined in `tutorial/dataset/*.yaml`:

```yaml
case:
  id: semantic_match_electronics
  description: Test semantic equivalence - Electronics matches Electronics & Technology

input: {}  # Function takes no input

run:
  kind: python
  target: tutorial.product_categories.get_product_categories
  timeout_ms: 5000

asserts:
  - path: $
    op: llm_judge
    config:
      ground_truth: "Electronics"
      prompt: "Does this list of categories contain an item semantically equivalent to '{ground_truth}'? Consider semantic meaning, not exact wording."
```

**Section breakdown:**

1. **case**: Metadata about the test
   - `id`: Unique identifier
   - `description`: Human-readable explanation

2. **input**: Data to pass to the function
   - Empty `{}` because `get_product_categories()` takes no arguments

3. **run**: How to execute the function
   - `kind: python`: Execute a Python function
   - `target`: Module path to the function
   - `timeout_ms`: Maximum execution time

4. **asserts**: Validation rules to check
   - `path: $`: JSONPath selector ($ = entire result)
   - `op: llm_judge`: Use LLM-based semantic matching
   - `config`: Operator-specific configuration
     - `ground_truth`: What to compare against
     - `prompt`: Optional custom prompt template

#### Important Data Serialization Note

**The function returns `list[str]`**, not a dict. When the Engine evaluates:
- JSONPath `$` returns: `["Electronics & Technology", "Home & Kitchen", ...]`
- llm_judge serializes as: `{"value": ["Electronics & Technology", ...]}`
- **LLM receives a JSON array**, not comma-separated text

This differs from direct Python usage where you might join the list into a string. The Engine approach is more structured and production-ready.

#### Available Test Cases

1. **01-semantic-match.yaml** - Semantic equivalence (Electronics → Electronics & Technology)
2. **02-exact-match.yaml** - Exact string matching
3. **03-negative-case.yaml** - Non-existent category (should fail)
4. **04-variant-wording.yaml** - Word order independence
5. **05-custom-prompt.yaml** - Custom prompt with system instructions

### Environment Configuration

The `tutorial/ollama.env` file sets required environment variables:

```bash
# API endpoint for local Ollama instance
export JUDGE_LLM_ENDPOINT=http://localhost:11434/v1

# Model to use for semantic validation
export JUDGE_LLM_MODEL=llama3.2:latest

# API key (required by client, but Ollama doesn't validate it)
export JUDGE_LLM_API_KEY=ollama

# Timeout for LLM requests (seconds)
export JUDGE_LLM_TIMEOUT=60
```

**Why these variables?**
- `JUDGE_LLM_ENDPOINT`: Points to Ollama's OpenAI-compatible API
- `JUDGE_LLM_MODEL`: Specifies which Ollama model to use
- `JUDGE_LLM_API_KEY`: Required by OpenAI client (any value works for Ollama)
- `JUDGE_LLM_TIMEOUT`: Prevents hanging on slow responses

## Understanding the Output

When you run `./tutorial/run_tutorial.sh`, you'll see:

```
==========================================
 llm_judge Tutorial Runner
==========================================

➜ Checking Ollama availability...
✓ Ollama is running and accessible

➜ Loading environment configuration...
==========================================
 LLM Configuration Loaded
==========================================
Endpoint:  http://localhost:11434/v1
Model:     llama3.2:latest
Timeout:   60s
==========================================

➜ Validating environment variables...
✓ JUDGE_LLM_ENDPOINT is set
✓ JUDGE_LLM_MODEL is set
✓ JUDGE_LLM_API_KEY is set

==========================================
 Executing Tests
==========================================

Command: uv run pytest tutorial/test_product_categories.py -v

tutorial/test_product_categories.py::TestProductCategoriesSemanticValidation::test_electronics_category_exists_semantically PASSED [ 20%]
tutorial/test_product_categories.py::TestProductCategoriesSemanticValidation::test_exact_match_also_works PASSED [ 40%]
tutorial/test_product_categories.py::TestProductCategoriesSemanticValidation::test_semantic_mismatch_detected PASSED [ 60%]
tutorial/test_product_categories.py::TestProductCategoriesSemanticValidation::test_variant_wording_recognized PASSED [ 80%]
tutorial/test_product_categories.py::TestProductCategoriesSemanticValidation::test_custom_system_prompt_works PASSED [100%]

==========================================
 Tutorial Completed Successfully
==========================================
✓ All tests passed!
```

**What each test demonstrates:**

1. **test_electronics_category_exists_semantically**: Core semantic matching
   - Searches for "Electronics"
   - Finds "Electronics & Technology"
   - Passes because semantically equivalent

2. **test_exact_match_also_works**: Baseline validation
   - Shows llm_judge handles exact matches too
   - Searches for "Toys & Games" (exact wording)
   - Confirms basic functionality

3. **test_semantic_mismatch_detected**: Negative testing
   - Searches for "Real Estate" (not in categories)
   - Correctly returns `ok=False`
   - Validates llm_judge doesn't false positive

4. **test_variant_wording_recognized**: Robustness check
   - Searches for "Clothing and Fashion" (reversed words)
   - Finds "Fashion & Clothing"
   - Demonstrates word-order independence

5. **test_custom_system_prompt_works**: Customization example
   - Shows how to provide custom system prompts
   - Demonstrates advanced usage patterns

## Customization

### Modify the Ground Truth

Test different semantic queries:

```python
# Try these ground truth values:
"Tech"           # Should match "Electronics & Technology"
"Beauty"         # Should match "Health & Beauty Products"
"Apparel"        # Should match "Fashion & Clothing"
"Literature"     # Should match "Books, Media & Entertainment"
```

### Customize Prompts

The `prompt` parameter accepts templates with placeholders:

```python
result = op_llm_judge(
    categories_text,
    {
        "ground_truth": "Electronics",
        "prompt": """
        Analyze if the categories list contains something
        semantically related to: {ground_truth}

        Categories: {actual}

        Consider synonyms, abbreviations, and related concepts.
        """,
    },
)
```

Available placeholders:
- `{actual}`: The selection being validated
- `{ground_truth}`: The expected value

### Customize System Prompt

For more control, customize the system prompt:

```python
result = op_llm_judge(
    categories_text,
    {
        "ground_truth": "Electronics",
        "system_prompt": """
        You are an expert in product categorization.
        Analyze if the actual data semantically matches the ground truth.
        Be strict about semantic equivalence - require true meaning alignment.
        """,
        "prompt": "Actual: {actual}\nExpected: {ground_truth}",
    },
)
```

### Test Different Categories

Modify `tutorial/product_categories.py` to test your own use case:

```python
def get_product_categories() -> list[str]:
    return [
        "Your Category 1",
        "Your Category 2",
        # ...
    ]
```

Then update tests to match your domain.

## Integration with Engine

This tutorial uses `op_llm_judge` directly. For production, integrate with the Engine and YAML:

### YAML Test Case Example

```yaml
case:
  id: product_category_validation
  description: Validate product categories semantically

input:
  # Input to your function

run:
  kind: python
  target: myapp.get_categories

asserts:
  - path: $  # Use JSONPath to extract categories
    op: llm_judge
    config:
      ground_truth: "Electronics"
      prompt: "Does the category list contain '{ground_truth}'?"
```

### Python Engine Usage

```python
from result_evaluator import Engine, load_test_case

# Load test case from YAML
test_case = load_test_case("test_categories.yaml")

# Run with Engine
engine = Engine()
result = engine.run_test(test_case)

# Check results
if result['status'] == 'PASS':
    print("✓ Categories validated semantically")
else:
    print(f"✗ Validation failed: {result['asserts']}")
```

See the main [README.md](README.md) for comprehensive Engine documentation.

## Next Steps

### Explore More Examples

- `tutorial/demo_standalone.py`: Standalone demo without pytest
- `tests/test_llm_operator.py`: Comprehensive llm_judge test suite
- `README.md`: Full operator documentation

### Learn Other Operators

- `sequence_in_order`: Validate ordered sequences in lists
- `contains`: Check for substring/element presence
- `match_regex`: Pattern matching
- `equals`: Strict equality

### Production Integration

1. **Switch to OpenAI for production**:
   ```bash
   unset JUDGE_LLM_ENDPOINT  # Use OpenAI default
   export JUDGE_LLM_MODEL=gpt-4o
   export JUDGE_LLM_API_KEY=sk-...  # Real OpenAI key
   ```

2. **Use in CI/CD pipelines**:
   - Set environment variables in CI
   - Run pytest as part of test suite
   - Monitor LLM API costs

3. **Optimize costs**:
   - Use llm_judge selectively (only where semantic validation needed)
   - Consider caching results for deterministic inputs
   - Use faster/cheaper models for simple validations

## Troubleshooting

### Ollama Not Running

**Error**: `Ollama is not running or not accessible`

**Solution**:
```bash
# Start Ollama server
ollama serve

# Or check if already running
ps aux | grep ollama

# Verify accessibility
curl http://localhost:11434/api/tags
```

### Model Not Found

**Error**: `Model 'llama3.2:latest' may not be pulled`

**Solution**:
```bash
# Pull the model
ollama pull llama3.2

# Verify it's available
ollama list

# Alternative: Use different model
export JUDGE_LLM_MODEL=llama2:latest
```

### Timeout Errors

**Error**: Test fails with timeout

**Solution**:
```bash
# Increase timeout (before running tests)
export JUDGE_LLM_TIMEOUT=120

# Or edit tutorial/ollama.env
# export JUDGE_LLM_TIMEOUT=120
```

### API Connection Errors

**Error**: `Connection refused` or `Cannot connect`

**Solution**:
1. Check Ollama is running: `ollama serve`
2. Verify endpoint is correct: `echo $JUDGE_LLM_ENDPOINT`
3. Test endpoint manually: `curl http://localhost:11434/api/tags`
4. Check firewall settings

### Environment Variables Not Set

**Error**: `Missing required environment variable`

**Solution**:
```bash
# Source the environment file
source tutorial/ollama.env

# Verify variables are set
echo $JUDGE_LLM_ENDPOINT
echo $JUDGE_LLM_MODEL
echo $JUDGE_LLM_API_KEY

# If still not set, check file exists
ls -la tutorial/ollama.env
```

### Tests Fail Unexpectedly

**Error**: Semantic match fails when it shouldn't

**Potential causes**:
1. Model not suitable for semantic matching (try llama3.2)
2. Prompt unclear (customize with more specific prompt)
3. Ground truth too ambiguous (be more specific)
4. Model output inconsistent (run test multiple times)

**Debug approach**:
```python
# Add debugging to see LLM reasoning
result = op_llm_judge(categories_text, {"ground_truth": "Electronics"})
print(f"Result: {result.ok}")
print(f"Message: {result.message}")  # LLM's reasoning
```

## Cost Considerations

`llm_judge` makes actual LLM API calls, which has cost implications:

### Ollama (Local) - FREE

- Runs on your machine
- No API costs
- Privacy: data stays local
- Slower than cloud APIs (depends on hardware)
- Requires model download (3-7 GB storage)

### OpenAI API - PAID

- Fast and accurate
- Costs per API call (see [OpenAI pricing](https://openai.com/pricing))
- Example: gpt-4o costs ~$0.005 per test with llm_judge
- For 1000 tests: ~$5
- Monitor costs in OpenAI dashboard

### Best Practices

1. **Use llm_judge selectively**: Only where semantic validation is truly needed
2. **Start with Ollama**: Develop and test locally for free
3. **Switch to OpenAI for production**: Better accuracy and reliability
4. **Consider caching**: Cache llm_judge results for deterministic inputs
5. **Monitor costs**: Track LLM API usage in production

## Summary

You've learned:
- ✓ What llm_judge is and when to use it
- ✓ How to configure Ollama for local testing
- ✓ How to write tests with semantic validation
- ✓ How to customize prompts for your domain
- ✓ How to integrate with Engine for production
- ✓ How to troubleshoot common issues

**Key takeaway**: Use `llm_judge` when semantic correctness matters more than exact wording. It makes your tests more robust and closer to human understanding.

Happy semantic testing!

## Related Documentation

- [README.md](README.md) - Full framework documentation
- [Ollama Documentation](https://ollama.ai/docs) - Ollama setup and usage
- [OpenAI API Docs](https://platform.openai.com/docs) - Production LLM integration
- [pytest Documentation](https://docs.pytest.org/) - Test framework reference
