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
        # Add more text samples as needed
    ]
})

# Function to process text in batches
def predict_text(df, batch_size=8):
    all_predictions = []
    for i in range(0, len(df), batch_size):
        batch_texts = df["text"].tolist()[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.tolist())
    return all_predictions

# Make predictions
Df['predictions'] = predict_text(Df, batch_size=8)

# Print the DataFrame with predictions
print(Df)
