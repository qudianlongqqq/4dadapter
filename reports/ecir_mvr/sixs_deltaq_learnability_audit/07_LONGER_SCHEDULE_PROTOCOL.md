# Conditional longer-full-objective protocol

This diagnostic is authorized only if both original last-250-step curves are clearly descending (at least 1% relative MAE reduction for each family) **and** head-only learnability passes. If authorized, it would use the same full v2 objective/model/32 records/seed with a predeclared maximum of 5,000 steps and a cosine horizon matched to that budget.

The observed original-curve classification is not written here. If either gate fails, `08_LONGER_SCHEDULE_RESULTS.csv` must record `NOT_RUN`; the audit must not launch it.
