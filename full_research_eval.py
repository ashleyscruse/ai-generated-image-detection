import torch
from transformers import pipeline
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# 1. Setup Model
print("Initializing Research Benchmark Model...")
detector = pipeline("image-classification", model="umm-maybe/AI-image-detector", device=0)

# 2. Define Data Sources
base_test_path = "/work/11110/michwusu520/ai-generated-image-detection/data/processed"
tasks = [
    ('test', 'real'), ('test', 'ai'),
    ('val', 'real'), ('val', 'ai')
]

all_data = []

# 3. Inference Loop
for split, category in tasks:
    path = os.path.join(base_test_path, split, category)
    files = [f for f in os.listdir(path) if f.endswith('.jpg')]
    
    print(f"Analyzing {split}/{category} ({len(files)} images)...")
    for filename in tqdm(files):
        try:
            img_path = os.path.join(path, filename)
            img = Image.open(img_path).convert("RGB")
            
            # Get full scores for AUC calculation
            outputs = detector(img)
            
            # Map outputs to a flat dict
            # Expecting [{'label': 'artificial', 'score': X}, {'label': 'human', 'score': Y}]
            scores = {item['label']: item['score'] for item in outputs}
            
            all_data.append({
                'filename': filename,
                'split': split,
                'ground_truth': category,
                'predicted_label': 'artificial' if scores['artificial'] > scores['human'] else 'real',
                'ai_score': scores['artificial'],
                'human_score': scores['human']
            })
        except Exception as e:
            continue

# 4. Save Raw Results for Analysis
df = pd.DataFrame(all_data)
df.to_csv("full_research_results.csv", index=False)
print("\n--- Raw Research Data Saved to full_research_results.csv ---")

# 5. Quick Summary Stats
df['correct'] = ((df['ground_truth'] == 'ai') & (df['predicted_label'] == 'artificial')) |                 ((df['ground_truth'] == 'real') & (df['predicted_label'] == 'real'))

accuracy = df['correct'].mean()
print(f"OVERALL BENCHMARK ACCURACY: {accuracy:.2%}")

ai_only = df[df['ground_truth'] == 'ai']
print(f"AI DETECTION RECALL: {ai_only['correct'].mean():.2%}")
