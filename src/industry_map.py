# Broad SIC division-based mapping (more complete than a small lookup).
# Reference: SEC SIC divisions grouped into standard industry buckets.

DEFAULT = "Unknown"

SIC_RANGES = [
    (1, 9, "Agriculture"),
    (10, 14, "Mining"),
    (15, 17, "Construction"),
    (20, 39, "Manufacturing"),
    (40, 49, "Transportation"),
    (50, 51, "Wholesale"),
    (52, 59, "Retail"),
    (60, 67, "Finance"),
    (70, 89, "Services"),
    (91, 97, "Public"),
    (99, 99, "Other"),
]


def map_sic_to_industry(sic: str) -> str:
    code = str(sic or "").strip()
    if not code:
        return DEFAULT
    try:
        prefix = int(code[:2])
    except ValueError:
        return DEFAULT
    for start, end, name in SIC_RANGES:
        if start <= prefix <= end:
            return name
    return DEFAULT
