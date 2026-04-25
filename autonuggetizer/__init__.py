"""
AutoNuggetizer Reproduction

Re-implementation of the AutoNuggetizer framework from:
  Pradeep et al. (2025). "The Great Nugget Recall: Automating Fact Extraction
  and RAG Evaluation with Large Language Models." SIGIR 2025.
  DOI: 10.1145/3726302.3730090

This code reproduces the three-step pipeline:
  1. Nugget Creation  — iterative extraction of atomic facts from contexts
  2. Nugget Scoring    — labeling each nugget as "vital" or "okay"
  3. Nugget Assignment — checking if nuggets appear in system answers

Usage:
  from autonuggetizer.pipeline import run_autonuggetizer, enable_verbose_logging
  enable_verbose_logging(True)   # toggle detailed intermediate output
"""
