"""
Referral-triage mapping module.

Converts the underlying 5-class DR severity output into the action-oriented
three-tier output (Normal / Monitor / Refer) described in Chapter 1.

*** IMPORTANT — matches Limitations, Item 3 in the manuscript ***
This mapping is a PROPOSED DESIGN, not yet clinically confirmed. It requires
review by a clinical adviser (see the USEP consultation request) before
being treated as final. Do not present this mapping as validated in your
defense — see Limitations for the exact wording to use if asked.
"""

# Proposed mapping — PENDING CLINICAL REVIEW.
TRIAGE_MAP = {
    "No_DR": "Normal",
    "Mild": "Monitor",
    "Moderate": "Monitor",
    "Severe": "Refer",
    "Proliferative_DR": "Refer",
}


def map_to_triage(dr_class: str) -> str:
    """
    Maps an underlying DR severity class to the simplified triage output.
    Defaults to "Refer" for any unrecognized class — when in doubt, the
    system should err toward flagging for human review, not silently
    passing as Normal.
    """
    return TRIAGE_MAP.get(dr_class, "Refer")
