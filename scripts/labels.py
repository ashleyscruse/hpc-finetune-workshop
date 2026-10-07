"""The six complaint categories the model learns, and how CFPB product names map to them.

CFPB has renamed its products over the years ("Credit reporting" became "Credit
reporting, credit repair services, or other personal consumer reports", and so on), so
the mapping matches on substrings rather than exact names.
"""

LABELS = [
    "Credit reporting",
    "Debt collection",
    "Mortgage",
    "Credit card",
    "Bank account",
    "Money transfer & payments",
]

LABEL2ID = {name: i for i, name in enumerate(LABELS)}


def product_to_label(product):
    """Return one of LABELS for a raw CFPB product name, or None to drop the row."""
    if not isinstance(product, str):
        return None
    p = product.lower()
    if "credit reporting" in p:
        return "Credit reporting"
    if "debt collection" in p:
        return "Debt collection"
    if "mortgage" in p:
        return "Mortgage"
    if p.startswith("credit card"):
        return "Credit card"
    if "checking or savings" in p or "bank account" in p:
        return "Bank account"
    if "money transfer" in p or "virtual currency" in p or p == "prepaid card":
        return "Money transfer & payments"
    return None
