import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# Path to the checkpoint
checkpoint_path = 'path/to/your/checkpoint'

# Load the tokenizer and model from the checkpoint
tokenizer = BertTokenizer.from_pretrained(checkpoint_path)
model = BertForSequenceClassification.from_pretrained(checkpoint_path)

# Assume Df is your DataFrame containing the text column
Df = pd.DataFrame({
    'text': [
        "Sample text 1",
        "Sample text 2",
        "Sample text 3",
    ]
})

# Preprocess the text data
inputs = tokenizer(Df["text"].tolist(), padding=True, truncation=True, return_tensors="pt")

# Make predictions
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get predicted class (for classification)
predictions = torch.argmax(logits, dim=-1)

# Convert predictions to a list and add to the DataFrame
Df['predictions'] = predictions.tolist()

# Print the DataFrame with predictions
print(Df)
