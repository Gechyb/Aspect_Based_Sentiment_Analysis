import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocess import normalize_text, to_bio_sentiment

sentence = "The   battery life is great"
sent = normalize_text(sentence)  # "The battery life is great"
tokens = sent.split()  # ["The", "battery", "life", "is", "great"]

# Step 1: start with no aspects
annotated_tokens = [{"token": t, "aspect": False} for t in tokens]

# Step 2: mark "battery life" as a POS aspect
aspect_tokens = ["battery", "life"]
L = len(aspect_tokens)

for i in range(len(tokens) - L + 1):
    if tokens[i : i + L] == aspect_tokens:
        # mark battery
        annotated_tokens[i]["aspect"] = True
        annotated_tokens[i]["sent"] = "POS"
        annotated_tokens[i]["begin"] = True
        # mark life
        annotated_tokens[i + 1]["aspect"] = True
        annotated_tokens[i + 1]["sent"] = "POS"
        annotated_tokens[i + 1]["begin"] = False

labels = to_bio_sentiment(annotated_tokens)
print(tokens)
print(labels)
