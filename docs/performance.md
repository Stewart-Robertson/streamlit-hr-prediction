# **Model performance**

## ROC curve
- Shows how well the model separates leavers vs stayers over all cut-offs.
- **AUC** close to 1.0 is great; ~0.65–0.75 is common for HR attrition.
- Use it to pick reasonable thresholds, not to claim perfection.

## Lift by decile
- Ranks employees by risk, splits into 10 equal buckets.
- **Lift** > 2 in the top decile means the model is >2x more likely to flag a leaver than random → excellent for targeting.
- This model is approximately 3x more likely to flag a leaver than random in the top decile.