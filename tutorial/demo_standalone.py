#!/usr/bin/env python3
"""Standalone demo: Product category validation with llm_judge.

This script demonstrates semantic matching without requiring pytest.
Perfect for quick exploration or integration into other scripts.

Prerequisites:
    - Ollama running locally (ollama serve)
    - Model pulled (ollama pull llama3.2)
    - Environment configured (source tutorial/ollama.env)

Run with:
    python tutorial/demo_standalone.py
"""

import os
import sys

# Add project root to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from result_evaluator.runtime.operators import op_llm_judge
from tutorial.product_categories import get_product_categories


def check_environment() -> bool:
    """Verify environment variables are configured.

    Returns:
        True if environment is configured, False otherwise.
    """
    required_vars = ["JUDGE_LLM_API_KEY", "JUDGE_LLM_MODEL", "JUDGE_LLM_ENDPOINT"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("\nPlease run: source tutorial/ollama.env")
        return False

    return True


def print_separator() -> None:
    """Print visual separator."""
    print("\n" + "=" * 70 + "\n")


def demo_basic_semantic_matching() -> None:
    """Demonstrate basic semantic matching."""
    print("DEMO 1: Basic Semantic Matching")
    print("-" * 70)

    # Get categories
    categories = get_product_categories({})
    print(f"\nProduct categories ({len(categories)} total):")
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat}")

    # Test semantic match
    print("\nSearching for: 'Electronics'")
    print("Actual category: 'Electronics & Technology'")

    categories_text = ", ".join(categories)
    result = op_llm_judge(
        categories_text,
        {
            "ground_truth": "Electronics",
            "prompt": "Does the actual text contain a semantically equivalent category to '{ground_truth}'?",
        },
    )

    if result.ok:
        print("\n✓ SUCCESS: Semantic match found!")
        print("  llm_judge recognized 'Electronics & Technology' matches 'Electronics'")
    else:
        print(f"\n✗ FAILED: {result.message}")


def demo_exact_matching() -> None:
    """Demonstrate exact matching also works."""
    print("DEMO 2: Exact Matching")
    print("-" * 70)

    categories = get_product_categories({})
    categories_text = ", ".join(categories)

    print("\nSearching for: 'Toys & Games' (exact wording)")

    result = op_llm_judge(
        categories_text,
        {"ground_truth": "Toys & Games"},
    )

    if result.ok:
        print("\n✓ SUCCESS: Exact match works too!")
    else:
        print(f"\n✗ FAILED: {result.message}")


def demo_missing_category() -> None:
    """Demonstrate detection of missing categories."""
    print("DEMO 3: Missing Category Detection")
    print("-" * 70)

    categories = get_product_categories({})
    categories_text = ", ".join(categories)

    print("\nSearching for: 'Real Estate' (not in categories)")

    result = op_llm_judge(
        categories_text,
        {"ground_truth": "Real Estate"},
    )

    if not result.ok:
        print("\n✓ SUCCESS: Correctly identified missing category!")
        print(f"  LLM reasoning: {result.message}")
    else:
        print("\n✗ UNEXPECTED: Found match when none should exist")


def demo_variant_wording() -> None:
    """Demonstrate matching with variant wording."""
    print("DEMO 4: Variant Wording")
    print("-" * 70)

    categories = get_product_categories({})
    categories_text = ", ".join(categories)

    print("\nSearching for: 'Clothing and Fashion' (word order reversed)")
    print("Actual category: 'Fashion & Clothing'")

    result = op_llm_judge(
        categories_text,
        {"ground_truth": "Clothing and Fashion"},
    )

    if result.ok:
        print("\n✓ SUCCESS: Semantic match despite different word order!")
    else:
        print(f"\n✗ FAILED: {result.message}")


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print(" llm_judge Semantic Matching Demo")
    print("=" * 70)

    # Check environment
    if not check_environment():
        sys.exit(1)

    print("\n✓ Environment configured")
    print(f"  Endpoint: {os.getenv('JUDGE_LLM_ENDPOINT')}")
    print(f"  Model: {os.getenv('JUDGE_LLM_MODEL')}")

    # Run demos
    print_separator()
    demo_basic_semantic_matching()

    print_separator()
    demo_exact_matching()

    print_separator()
    demo_missing_category()

    print_separator()
    demo_variant_wording()

    print_separator()
    print("✓ All demonstrations completed!")
    print("\nNext steps:")
    print("  - Explore tutorial/test_product_categories.py for pytest version")
    print("  - Read TUTORIAL.md for comprehensive guide")
    print("  - Check README.md for full operator documentation")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
