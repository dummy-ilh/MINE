import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# Path to the checkpoint
checkpoint_path = 'path/to/your/checkpoint'

# Load the tokenizer and model from the checkpoint
tokenizer = BertTokenizer.from_pretrained(checkpoint_path)
model = BertForSequenceClassification.from_pretrained(checkpoint_path)

# Sample DataFrame containing text column
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
    num_samples = len(df)
    print(f"Total samples: {num_samples}")

    for i in range(0, num_samples, batch_size):
        end_index = min(i + batch_size, num_samples)
        print(f"Processing batch {i // batch_size + 1}, indexes {i} to {end_index - 1}")

        batch_texts = df["text"].tolist()[i:end_index]
        if len(batch_texts) == 0:
            print(f"No texts in batch {i // batch_size + 1}")
            continue

        try:
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.tolist())
        except Exception as e:
            print(f"Error
