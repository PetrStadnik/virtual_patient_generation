import simple_icd_10 as icd
from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────
#  ICD-10 WHO lookup result
# ──────────────────────────────────────────────

@dataclass
class ICDDiagnosisInfo:
    code: str
    valid: bool                         # Exists in ICD-10 WHO?
    description: Optional[str]          # Diagnosis name
    billable: Optional[bool]            # Is it a leaf (most precious)?
    chapter: Optional[str]              # Chapter e.g. "IX"
    chapter_description: Optional[str]  # Chapter description
    block: Optional[str]                # Block, e.g. "I20-I25"
    block_description: Optional[str]    # Block description
    parent: Optional[str]               # Direct parent, e.g. "I21" for "I21.0"
    parent_description: Optional[str]   # Parent description
    children: list[str]                 # Subcodes (empty if is a leaf code)
    ancestors: list[str]                # Whole path from parent to chapter


# ──────────────────────────────────────────────
#  Methods
# ──────────────────────────────────────────────

def lookup(code: str) -> ICDDiagnosisInfo:
    normalized = code.strip().upper()

    if not icd.is_valid_item(normalized):
        return ICDDiagnosisInfo(
            code=normalized,
            valid=False,
            description=None,
            billable=None,
            chapter=None,
            chapter_description=None,
            block=None,
            block_description=None,
            parent=None,
            parent_description=None,
            children=[],
            ancestors=[],
        )

    ancestors = icd.get_ancestors(normalized)
    chapter = ancestors[-1] if ancestors else None
    block   = ancestors[-2] if len(ancestors) >= 2 else None
    parent  = ancestors[0]  if ancestors else None

    _roman_chars = set("IVXLCDM")
    if parent and ("-" in parent or all(c in _roman_chars for c in parent)):
        parent = None

    return ICDDiagnosisInfo(
        code=normalized,
        valid=True,
        description=icd.get_description(normalized),
        billable=icd.is_leaf(normalized),
        chapter=chapter,
        chapter_description=icd.get_description(chapter) if chapter else None,
        block=block,
        block_description=icd.get_description(block) if block else None,
        parent=parent,
        parent_description=icd.get_description(parent) if parent else None,
        children=icd.get_children(normalized),
        ancestors=ancestors,
    )


# ──────────────────────────────────────────────
#  Validation for FHIR Condition resource
# ──────────────────────────────────────────────

def validate_icd_for_fhir(code: str) -> ICDDiagnosisInfo:
    info = lookup(code)

    # check if code exists
    if not info.valid:
        print(f"Code '{code}' does not exist in ICD-10 WHO. ")
        print(f"Check the code, do not use ICD-10-CM.")

    # check if it is a leaf
    elif not info.billable:
        children_str = ", ".join(info.children)
        print(f"Code '{code}' ({info.description}) is a category – not precious enough. ")
        print(f"Select one of subcodes: {children_str}")

    return info


# ──────────────────────────────────────────────
#  Helper: subcodes tree
# ──────────────────────────────────────────────

def print_tree(code: str, indent: int = 0) -> None:
    """
    Recursively prints the subtree of child codes for a given ICD-10 code.
    Useful for debugging or understanding code granularity.
    """
    normalized = code.strip().upper()
    if not icd.is_valid_item(normalized):
        print(f"{'  ' * indent}[invalid code: {normalized}]")
        return

    desc      = icd.get_description(normalized)
    is_leaf   = icd.is_leaf(normalized)
    leaf_mark = " ✓" if is_leaf else ""
    print(f"{'  ' * indent}{normalized}  –  {desc}{leaf_mark}")

    for child in icd.get_children(normalized):
        print_tree(child, indent + 1)


# ──────────────────────────────────────────────
#  Demo
# ──────────────────────────────────────────────

if __name__ == "__main__":

 print(lookup("A09.9"))