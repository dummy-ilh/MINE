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

# Function to process text in batches and make predictions
def predict_text(df, batch_size=8):
    all_predictions = []
    all_scores = []
    all_labels = []
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
                predictions = torch.argmax(logits, dim=-1).tolist()
                scores = torch.softmax(logits, dim=1).max(dim=1).values.tolist()
                labels = [model.config.id2label[pred] for pred in predictions]

                all_predictions.extend(predictions)
                all_scores.extend(scores)
                all_labels.extend(labels)
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            break

        # Additional debug: print batch predictions
        print(f"Batch {i // batch_size + 1} predictions: {predictions}")
        print(f"Batch {i // batch_size + 1} scores: {scores}")
        print(f"Batch {i // batch_size + 1} labels: {labels}")

    return all_predictions, all_scores, all_labels

# Make predictions
predictions, scores, labels = predict_text(Df, batch_size=2)
if predictions:
    Df['predictions'] = predictions
    Df['scores'] = scores
    Df['labels'] = labels
    # Print the DataFrame with predictions
    print(Df)
else:
    print("No predictions were made.")
