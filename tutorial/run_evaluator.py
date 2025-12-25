"""Tutorial runner for result-evaluator using YAML test cases.

This script demonstrates production-style usage of the result_evaluator Engine
with declarative YAML test cases and structured JSONL logging.

Usage:
    python tutorial/run_evaluator.py                    # Run all tests
    python tutorial/run_evaluator.py --dataset "01-*.yaml"  # Run filtered tests
"""

import argparse
import logging
import sys
from pathlib import Path

from result_evaluator import Engine, load_test_case
from tutorial.jsonl_formatter import JSONLFormatter

# Load environment variables (optional - can be sourced externally)
try:
    from dotenv import load_dotenv

    load_dotenv("tutorial/ollama.env")
except ImportError:
    # dotenv not available - assume environment is already configured
    pass

# Define evaluation directory
EVAL_DIR = Path(__file__).parent / "dataset"
CURRENT_DIR = Path(__file__).parent

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Run evaluation tests on product categories with llm_judge"
)
parser.add_argument(
    "--dataset",
    type=str,
    help="Dataset file mask (glob pattern, e.g., '01-*.yaml'). If not specified, all YAML files will be processed.",
)
args = parser.parse_args()

# Setup dual output: stdout + file
# Create JSONL formatter
jsonl_formatter = JSONLFormatter()

# StreamHandler for stdout - INFO level (clean console)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(jsonl_formatter)

# FileHandler for evaluation_log.jsonl (overwrite mode)
file_handler = logging.FileHandler(
    CURRENT_DIR / "evaluation_log.jsonl", mode="w", encoding="utf-8"
)
file_handler.setFormatter(jsonl_formatter)

# Configure root logger - DEBUG level to capture LLM prompts
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers.clear()
root_logger.addHandler(stream_handler)
root_logger.addHandler(file_handler)

# Get logger for this module
logger = logging.getLogger(__name__)

# Ensure result_evaluator loggers capture DEBUG messages
logging.getLogger("result_evaluator").setLevel(logging.DEBUG)
logging.getLogger("result_evaluator.runtime.llm").setLevel(logging.DEBUG)

# Find dataset files based on command line arguments
if not EVAL_DIR.exists():
    logger.error("Dataset directory not found", extra={"path": str(EVAL_DIR)})
    sys.exit(1)

if args.dataset:
    # Use specified mask
    dataset_files = sorted(EVAL_DIR.glob(args.dataset))
    if not dataset_files:
        logger.error("No dataset files found", extra={"pattern": args.dataset})
        sys.exit(1)
else:
    # Find all dataset files (both YAML and YML)
    yaml_files = list(EVAL_DIR.glob("*.yaml"))
    yml_files = list(EVAL_DIR.glob("*.yml"))
    dataset_files = sorted(yaml_files + yml_files)

    if not dataset_files:
        logger.error(
            "No dataset files found",
            extra={"patterns": ["*.yaml", "*.yml"], "directory": str(EVAL_DIR)},
        )
        sys.exit(1)

# Run all tests
engine = Engine()
results = []

for dataset_file in dataset_files:
    logger.info("Test started", extra={"dataset": dataset_file.name})

    try:
        test_case = load_test_case(dataset_file)
        result = engine.run_test(test_case)
        results.append(result)

        # Log test completion with results
        log_data = {
            "dataset": dataset_file.name,
            "case_id": result.get("case_id", "unknown"),
            "status": result.get("status", "ERROR"),
        }

        # Add asserts only if present
        if "asserts" in result:
            log_data["asserts"] = result["asserts"]

        # Add error message if present
        if "error" in result:
            log_data["error"] = result["error"]

        logger.info("Test completed", extra=log_data)

    except Exception as e:
        logger.error(
            "Failed to load or run test",
            extra={"dataset": dataset_file.name, "error": str(e)},
            exc_info=True,
        )
        results.append(
            {
                "case_id": dataset_file.stem,
                "status": "ERROR",
                "error": f"Failed to load/run test: {str(e)}",
            }
        )

# Log summary
status_counts = {"PASS": 0, "FAIL": 0, "ERROR": 0}
for result in results:
    status = result.get("status", "ERROR")
    status_counts[status] = status_counts.get(status, 0) + 1

logger.info(
    "All tests completed",
    extra={
        "total": len(results),
        "passed": status_counts["PASS"],
        "failed": status_counts["FAIL"],
        "errors": status_counts["ERROR"],
        "results": [
            {"case_id": r.get("case_id", "unknown"), "status": r.get("status", "ERROR")}
            for r in results
        ],
    },
)

# Exit with appropriate code
if status_counts["ERROR"] > 0:
    sys.exit(2)  # Errors occurred
elif status_counts["FAIL"] > 0:
    sys.exit(1)  # Tests failed
else:
    sys.exit(0)  # All tests passed
