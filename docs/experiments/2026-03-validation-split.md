# March Validation Split Rationale

This note records the validation assumptions used for the carbon tower outlet temperature forecasting work.

- Keep chronological order when building training and validation windows.
- Avoid random row shuffling because adjacent sensor readings share process-state information.
- Reserve the latest stable window for validation so drift and delayed response are visible.
- Compare rolling validation scores rather than relying on a single holdout score.

The main risk is leakage from future process measurements into features that should only use historical context.
