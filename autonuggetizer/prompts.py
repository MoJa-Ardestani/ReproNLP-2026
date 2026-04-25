"""
Exact prompts from the Nugget paper (Pradeep et al., SIGIR 2025).

Source: Papers_latex/nugget/prompts/
  - prompt_iter_nuggetizer.tex  (Figure 1 in paper)
  - prompt_nugget_scorer.tex    (Figure 2 in paper)
  - prompt_nugget_assigner.tex  (Figure 3 in paper)
"""

# ---------------------------------------------------------------------------
# Prompt 1 — Iterative Nugget Creation (Figure 1)
# Called repeatedly with batches of 10 context segments.
# Each turn updates the running nugget list.
# ---------------------------------------------------------------------------

NUGGET_CREATION_SYSTEM = (
    "You are NuggetizeLLM, an intelligent assistant that can update "
    "a list of atomic nuggets to best provide all the information "
    "required for the query."
)

NUGGET_CREATION_USER = """\
Update the list of atomic nuggets of information (1-12 words), if needed, \
so they best provide the information required for the query. Leverage only \
the initial list of nuggets (if exists) and the provided context (this is \
an iterative process). Return only the final list of all nuggets in a \
Pythonic list format (even if no updates). Make sure there is no redundant \
information. Ensure the updated nugget list has at most 30 nuggets (can be \
less), keeping only the most vital ones. Order them in decreasing order of \
importance. Prefer nuggets that provide more interesting information.

Search Query: {query}

Context:
{context_block}

Search Query: {query}

Initial Nugget List: {nugget_list}

Initial Nugget List Length: {nugget_list_length}

Only update the list of atomic nuggets (if needed, else return as is). \
Do not explain. Always answer in short nuggets (not questions). List in \
the form ["a", "b", ...] and a and b are strings with no mention of ".\

Updated Nugget List:"""


# ---------------------------------------------------------------------------
# Prompt 2 — Nugget Importance Scoring (Figure 2)
# Labels each nugget as "vital" or "okay".
# At each turn, at most 10 nuggets are passed.
# ---------------------------------------------------------------------------

NUGGET_SCORING_SYSTEM = (
    "You are NuggetizeScoreLLM, an intelligent assistant that can label "
    "a list of atomic nuggets based on their importance for a given "
    "search query."
)

NUGGET_SCORING_USER = """\
Based on the query, label each of the {num_nuggets} nuggets either a vital \
or okay based on the following criteria. Vital nuggets represent concepts \
that must be present in a "good" answer; on the other hand, okay nuggets \
contribute worthwhile information about the target but are not essential. \
Return the list of labels in a Pythonic list format (type: List[str]). The \
list should be in the same order as the input nuggets. Make sure to provide \
a label for each nugget.

Search Query: {query}

Nugget List: {nugget_list}

Only return the list of labels (List[str]). Do not explain.

Labels:"""


# ---------------------------------------------------------------------------
# Prompt 3 — Nugget Assignment (Figure 3)
# Checks whether each nugget is captured by a given passage (system answer).
# At each turn, at most 10 nuggets are passed.
# ---------------------------------------------------------------------------

NUGGET_ASSIGNMENT_SYSTEM = (
    "You are NuggetizeAssignerLLM, an intelligent assistant that can label "
    "a list of atomic nuggets based on if they are captured by a given passage."
)

NUGGET_ASSIGNMENT_USER = """\
Based on the query and passage, label each of the {num_nuggets} nuggets \
either as support, partial_support, or not_support using the following \
criteria. A nugget that is fully captured in the passage should be labeled \
as support. A nugget that is partially captured in the passage should be \
labeled as partial_support. If the nugget is not captured at all, label it \
as not_support. Return the list of labels in a Pythonic list format (type: \
List[str]). The list should be in the same order as the input nuggets. Make \
sure to provide a label for each nugget.

Search Query: {query}
Passage: {passage}
Nugget List: {nugget_list}

Only return the list of labels (List[str]). Do not explain.

Labels:"""
