#!/usr/bin/env bash
# Tutorial runner script with comprehensive validation
#
# This script validates the environment and runs the llm_judge tutorial tests.
# It checks prerequisites before execution to provide helpful error messages.
#
# Usage:
#   ./tutorial/run_tutorial.sh                    # Run all tests
#   ./tutorial/run_tutorial.sh "01-*.yaml"        # Run specific test(s)
#   ./tutorial/run_tutorial.sh dataset/01-semantic-match.yaml  # Run single test

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Script directory (absolute path)
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo -e "${BLUE}"
    echo "=========================================="
    echo " $1"
    echo "=========================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "\n${BLUE}➜ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

# ==============================================================================
# Validation Functions
# ==============================================================================

check_ollama_running() {
    print_step "Checking Ollama availability..."

    if ! command -v curl &> /dev/null; then
        print_error "curl is not installed (required for Ollama health check)"
        echo "Install curl or skip this check by running tests manually"
        return 1
    fi

    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_error "Ollama is not running or not accessible at http://localhost:11434"
        echo ""
        echo "To fix this:"
        echo "  1. Install Ollama from: https://ollama.ai"
        echo "  2. Start Ollama with: ollama serve"
        echo "  3. Or check if Ollama is running: ps aux | grep ollama"
        return 1
    fi

    print_success "Ollama is running and accessible"
    return 0
}

load_environment() {
    print_step "Loading environment configuration..."

    local env_file="$SCRIPT_DIR/ollama.env"

    if [[ ! -f "$env_file" ]]; then
        print_error "Environment file not found: $env_file"
        return 1
    fi

    # Source the environment file
    # shellcheck source=/dev/null
    source "$env_file"

    print_success "Environment loaded from ollama.env"
    return 0
}

validate_environment_variables() {
    print_step "Validating environment variables..."

    local required_vars=("JUDGE_LLM_ENDPOINT" "JUDGE_LLM_MODEL" "JUDGE_LLM_API_KEY")
    local missing_vars=()

    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        else
            print_success "$var is set: ${!var}"
        fi
    done

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        print_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        echo ""
        echo "Ensure tutorial/ollama.env contains all required variables"
        return 1
    fi

    return 0
}

check_model_availability() {
    print_step "Checking if model is available in Ollama..."

    if ! command -v curl &> /dev/null; then
        print_warning "Cannot check model availability (curl not installed)"
        return 0  # Don't fail, just warn
    fi

    local model_name="${JUDGE_LLM_MODEL}"

    if curl -s http://localhost:11434/api/tags | grep -q "$model_name"; then
        print_success "Model '$model_name' is available"
    else
        print_warning "Model '$model_name' may not be pulled"
        echo ""
        echo "To pull the model, run:"
        echo "  ollama pull $model_name"
        echo ""
        echo "Continuing anyway (this might fail if model is truly missing)..."
    fi

    return 0
}

run_tutorial_tests() {
    local dataset_mask="$1"

    print_step "Running tutorial tests..."

    cd "$PROJECT_ROOT"

    # Check if run_evaluator.py exists
    if [[ ! -f "tutorial/run_evaluator.py" ]]; then
        print_error "tutorial/run_evaluator.py not found"
        return 1
    fi

    # Build command with optional dataset filter
    local cmd="PYTHONPATH=. uv run python tutorial/run_evaluator.py"
    if [[ -n "$dataset_mask" ]]; then
        cmd="$cmd --dataset \"$dataset_mask\""
    fi

    echo ""
    echo "Command: $cmd"
    echo ""

    # Run the evaluator with YAML test cases
    if [[ -n "$dataset_mask" ]]; then
        if PYTHONPATH=. uv run python tutorial/run_evaluator.py --dataset "$dataset_mask"; then
            return 0
        else
            return 1
        fi
    else
        if PYTHONPATH=. uv run python tutorial/run_evaluator.py; then
            return 0
        else
            return 1
        fi
    fi
}

# ==============================================================================
# Main Execution
# ==============================================================================

main() {
    local dataset_mask="${1:-}"

    print_header "llm_judge Tutorial Runner"

    # Show dataset filter if provided
    if [[ -n "$dataset_mask" ]]; then
        echo ""
        echo "Dataset filter: $dataset_mask"
        echo ""
    fi

    # Step 1: Check Ollama
    if ! check_ollama_running; then
        exit 1
    fi

    # Step 2: Load environment
    if ! load_environment; then
        exit 1
    fi

    # Step 3: Validate environment variables
    if ! validate_environment_variables; then
        exit 1
    fi

    # Step 4: Check model (non-fatal)
    check_model_availability

    # Step 5: Run tests
    echo ""
    print_header "Executing Tests"

    if run_tutorial_tests "$dataset_mask"; then
        echo ""
        print_header "Tutorial Completed Successfully"
        echo ""
        print_success "All tests passed!"        
        
        exit 0
    else
        echo ""
        print_error "Tutorial tests failed"
        echo ""
        echo "Troubleshooting:"
        echo "  - See the evaluation_log.jsonl"        
        echo ""
        exit 1
    fi
}

# Run main function with optional dataset mask argument
main "$@"
