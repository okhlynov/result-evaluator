"""Product categorization function for tutorial demonstration."""


from typing import Any


def get_product_categories(_: dict[str,Any]) -> list[str]:
    """Return list of product categories for e-commerce platform.

    This function returns a fixed list of product categories with semantic
    variations that demonstrate the value of llm_judge for semantic matching.

    For example, "Electronics & Technology" would match semantic queries for:
    - "Electronics"
    - "Tech Products"
    - "Electronic Devices"

    Args:
        input_data: Input data dict (unused - function returns fixed list).

    Returns:
        List of product category names.

    Example:
        >>> categories = get_product_categories({})
        >>> len(categories)
        8
        >>> "Electronics & Technology" in categories
        True
    """
    return [
        "Electronics & Technology",
        "Home & Kitchen Appliances",
        "Fashion & Clothing",
        "Books, Media & Entertainment",
        "Sports & Outdoor Equipment",
        "Health & Beauty Products",
        "Toys & Games",
        "Automotive & Tools",
    ]
