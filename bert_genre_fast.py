import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. LOAD DATA
print("\n" + "="*70)
print("STEP 1: LOADING DATA")
print("="*70)

csv_path = os.path.expanduser('~/Downloads')
train_df = pd.read_csv(os.path.join(csv_path, 'train.csv'))
test_df = pd.read_csv(os.path.join(csv_path, 'test.csv'))
genres_df = pd.read_csv(os.path.join(csv_path, 'movies_genres.csv'))

# USE SAMPLE FOR FASTER TRAINING (can increase later)
train_sample = train_df.sample(n=2000, random_state=42)  # 2000 samples instead of 8000
print(f" Train set (sampled): {len(train_sample)} samples")
print(f" Test set: {len(test_df)} samples")

# 2. PREPARE DATA
print("\n" + "="*70)
print("STEP 2: PREPARING DATA")
print("="*70)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Parse genre IDs
train_sample['genre_ids'] = train_sample['genre_ids'].apply(eval)
mlb = MultiLabelBinarizer()
y_train_all = mlb.fit_transform(train_sample['genre_ids'])

print(f" Genres classes: {len(mlb.classes_)}")

# Split into train (70%), val (30%)
train_texts, val_texts, y_train, y_val, train_ids, val_ids = train_test_split(
    train_sample['overview'].values,
    y_train_all,
    train_sample.index,
    test_size=0.3,
    random_state=42
)

test_texts = test_df['overview'].values

print(f" Train: {len(train_texts)} samples")
print(f" Validation: {len(val_texts)} samples")
print(f" Test: {len(test_texts)} samples")


# 3. TOKENIZE DATA

print("\n" + "="*70)
print("STEP 3: TOKENIZING TEXT")
print("="*70)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = 96  # Shorter for faster training

def tokenize_texts(texts):
    encodings = tokenizer(
        texts.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encodings

print(" Tokenizing all datasets...")
train_encodings = tokenize_texts(train_texts)
val_encodings = tokenize_texts(val_texts)
test_encodings = tokenize_texts(test_texts)

# Create PyTorch datasets
class GenreDataset:
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = GenreDataset(train_encodings, y_train)
val_dataset = GenreDataset(val_encodings, y_val)
test_dataset = GenreDataset(test_encodings)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print(f" Tokenization complete")

#
# 4. LOAD MODEL
# 
print("\n" + "="*70)
print("STEP 4: LOADING MODEL")
print("="*70)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(mlb.classes_),
    problem_type="multi_label_classification"
)
model = model.to(device)

print(f" Model: {model_name}")
print(f" Number of labels: {len(mlb.classes_)}")


# 5. TRAINING SETUP

print("\n" + "="*70)
print("STEP 5: TRAINING SETUP")
print("="*70)

from torch.optim import AdamW
from sklearn.metrics import f1_score, hamming_loss

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 2  # Fewer epochs for speed

print(f"Optimizer: AdamW (lr=2e-5)")
print(f"Epochs: {num_epochs}")
print(f"Batch size: 32")


# 6. TRAINING LOOP

print("\n" + "="*70)
print("STEP 6: TRAINING (this will take ~5-10 minutes)")
print("="*70)

for epoch in range(num_epochs):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print('='*70)
    
    # ---- TRAIN ----
    print("Training")
    model.train()
    train_loss = 0
    train_preds = []
    train_true = []
    
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        logits = outputs.logits
        preds = (torch.sigmoid(logits) > 0.5).detach().cpu().numpy()
        train_preds.extend(preds)
        train_true.extend(labels.detach().cpu().numpy())
        
        if (i + 1) % max(1, len(train_loader) // 3) == 0:
            print(f"  Batch {i+1}/{len(train_loader)}")
    
    train_loss /= len(train_loader)
    train_preds = np.array(train_preds)
    train_true = np.array(train_true)
    
    train_f1 = f1_score(train_true, train_preds, average='micro', zero_division=0)
    train_hamming = hamming_loss(train_true, train_preds)
    
    # ---- VALIDATION ----
    print("Validating...")
    model.eval()
    val_loss = 0
    val_preds = []
    val_true = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            
            logits = outputs.logits
            preds = (torch.sigmoid(logits) > 0.5).detach().cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(labels.detach().cpu().numpy())
    
    val_loss /= len(val_loader)
    val_preds = np.array(val_preds)
    val_true = np.array(val_true)
    
    val_f1 = f1_score(val_true, val_preds, average='micro', zero_division=0)
    val_hamming = hamming_loss(val_true, val_preds)
    
    # Print results
    print(f"\n Results:")
    print(f"  Train Loss: {train_loss:.4f} | F1: {train_f1:.4f} | Hamming: {train_hamming:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | F1: {val_f1:.4f} | Hamming: {val_hamming:.4f}")


# 7. FINAL EVALUATION

print("\n" + "="*70)
print("STEP 7: FINAL EVALUATION ON ALL SETS")
print("="*70)

model.eval()

def evaluate_dataset(loader, dataset_labels, dataset_name):
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            preds.extend(pred)
            
            if 'labels' in batch:
                true_labels.extend(batch['labels'].cpu().numpy())
    
    preds = np.array(preds)
    true_labels = np.array(true_labels) if true_labels else None
    
    print(f"\n{'='*70}")
    print(f"📈 {dataset_name} SET RESULTS")
    print('='*70)
    
    if true_labels is not None:
        micro_f1 = f1_score(true_labels, preds, average='micro', zero_division=0)
        macro_f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
        hamming = hamming_loss(true_labels, preds)
        exact_match = (true_labels == preds).all(axis=1).sum() / len(true_labels)
        
        print(f"  F1-Score (Micro):      {micro_f1:.4f}")
        print(f"  F1-Score (Macro):      {macro_f1:.4f}")
        print(f"  Hamming Loss:          {hamming:.4f}")
        print(f"  Exact Match Accuracy:  {exact_match:.4f}")
    else:
        avg_genres = preds.sum(axis=1).mean()
        print(f"  Avg genres per sample: {avg_genres:.2f}")
        print(f"  Unique predictions:    {len(np.unique(preds, axis=0))}")
    
    return preds

# Evaluate
print("\nEvaluating TRAIN set...")
train_preds_final = evaluate_dataset(train_loader, y_train, "TRAIN")

print("\nEvaluating VALIDATION set...")
val_preds_final = evaluate_dataset(val_loader, y_val, "VALIDATION")

print("\nEvaluating TEST set...")
test_preds_final = evaluate_dataset(test_loader, None, "TEST")

# 8. GENERATE SUBMISSION

print("\n" + "="*70)
print("STEP 8: GENERATING SUBMISSION")
print("="*70)

test_predictions = []
for pred in test_preds_final:
    genre_indices = np.where(pred == 1)[0]
    genre_ids = [mlb.classes_[i] for i in genre_indices]
    
    if not genre_ids:
        # Most common genre
        genre_ids = [mlb.classes_[y_train.sum(axis=0).argmax()]]
    
    genre_str = ' '.join(map(str, sorted(genre_ids)))
    test_predictions.append(genre_str)

submission_df = pd.DataFrame({
    'movie_id': test_df['movie_id'],
    'genre_ids': test_predictions
})

output_path = os.path.join(csv_path, 'submission_bert.csv')
submission_df.to_csv(output_path, index=False)

print(f"✓ Submission saved: {output_path}")
print(f"\nFirst 10 predictions:")
print(submission_df.head(10))

# Validation
print("\n" + "="*70)
print("SUBMISSION VALIDATION")
print("="*70)

genre_counts = []
for genres_str in submission_df['genre_ids']:
    count = len(str(genres_str).split())
    genre_counts.append(count)

print(f" Total predictions: {len(submission_df)}")
print(f" Movies with 1 genre: {genre_counts.count(1)}")
print(f" Movies with 2 genres: {genre_counts.count(2)}")
print(f" Movies with 3+ genres: {sum(1 for c in genre_counts if c >= 3)}")
print(f" Average genres per movie: {np.mean(genre_counts):.2f}")


print(" BERT TRAINING COMPLETE!")
print("="*70)
print(f"\n Submission at: {output_path}")

