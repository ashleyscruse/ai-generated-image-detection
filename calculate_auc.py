import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv('full_research_results.csv')
# Ground truth: 1 for AI, 0 for Real
y_true = (df['ground_truth'] == 'ai').astype(int)
# The score the model gave for the 'artificial' label
y_scores = df['ai_score']

auc = roc_auc_score(y_true, y_scores)
print(f"\n--- RESEARCH METRIC ---")
print(f"Final AUC-ROC Score: {auc:.4f}")
if auc < 0.5:
    print("ANALYSIS: The detector is ANTI-CORRELATED. It is literally more likely to call a REAL image AI than an AI image AI.")
