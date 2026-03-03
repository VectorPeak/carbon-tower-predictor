# Lag Feature Candidates

Candidate features for outlet temperature prediction:

- Short lag windows for immediate sensor response: 1, 2, 3, and 5 steps.
- Medium lag windows for process inertia: 10, 15, and 30 steps.
- Rolling means and rolling standard deviations for feed stability.
- Difference features between current value and lagged value.
- Stability flags for periods with abrupt input changes.

These notes are intended to keep feature generation aligned with the physical delay of the carbon tower process.
