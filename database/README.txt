Multi-Disease Guidance Database - starter dataset

This is an educational starter knowledge base for the user's multi-disease ML application.
It is not a clinical guideline or a diagnostic/treatment system.

Files:
- diseases.csv: supported diseases and model input types
- guidance.csv: disease-specific educational guidance
- probability_rules.csv: display logic for model-estimated probability (not validated clinical risk thresholds)
- sources.csv: authoritative source references
- input_rules.csv: conservative input flags for discussion, not diagnoses

Important:
1. Keep ML prediction separate from medical guidance.
2. Do not use model probability as a clinical diagnosis or validated risk score.
3. Kidney guidance is intentionally individualized for potassium, phosphorus, protein, and fluids.
4. Expand and clinically review the database before public/clinical deployment.
