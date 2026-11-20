# DS9000 — Project

## Data Preprocessing

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = Path("deceptive-opinion.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True) # Create a folder for outputs
RANDOM_STATE = 9000 # Random seed
N_SPLITS = 5 # For 5-fold cross-validation

# Step 1: Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

# Data inspection
print("\nMissing values:")
print(df.isna().sum())

print("\nClass distribution:")
print(df["deceptive"].value_counts())

print("\nSentiment distribution:")
print(df["polarity"].value_counts())

print("\nNumber of hotels:", df["hotel"].nunique())

# Step 2: Check duplicates and label consistency
print("Duplicate reviews:", df.duplicated(subset=["text"]).sum()) # Check duplicate reviews

conflicts = df.groupby("text")["deceptive"].nunique() # Check conflicting labels
print("Conflicting labels:", (conflicts > 1).sum())

# Step 3: EDA
# Reviews by hotel
hotel_counts = df.groupby(["hotel", "deceptive"]).size().unstack(fill_value=0)
print(hotel_counts)
hotel_counts.plot(kind="bar", stacked=True, figsize=(12, 4), title="Reviews by hotel and label")
plt.xlabel("Hotel")
plt.ylabel("Number of reviews")
plt.tight_layout()
plt.show()

# Review length Distribution
df["word_count"] = df["text"].fillna("").str.split().str.len()
print(df.groupby("deceptive")["word_count"].describe().round(1))

plt.figure(figsize=(8, 4))
for label in ["truthful", "deceptive"]:
    plt.hist(df.loc[df["deceptive"] == label, "word_count"], bins=30, alpha=0.5, label=label)
plt.xlabel("Words per review")
plt.ylabel("Count")
plt.title("Review length distribution")
plt.legend()
plt.tight_layout()
plt.show()

# Step 4: Clean data and prepare labels
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Remove missing and empty reviews
clean_df = df.copy()
clean_df = clean_df.dropna(subset=["text", "deceptive", "hotel"])

clean_df["review_text"] = clean_df["text"].apply(clean_text)
clean_df = clean_df[clean_df["review_text"].str.len() > 0].copy()

# Encode labels
clean_df["label"] = clean_df["deceptive"].map({
    "truthful": 0,
    "deceptive": 1
})

# Validate labels and reset index
assert clean_df["label"].notna().all()
clean_df["label"] = clean_df["label"].astype(int)
clean_df = clean_df.reset_index(drop=True)

# Save cleaned data to outputs
clean_df.to_csv(OUTPUT_DIR / "cleaned_reviews.csv", index=False)

# Step 5: Hotel-grouped 5-fold cross-validation
# Prepare text, labels, and hotel groups
X_text = clean_df["review_text"]
y = clean_df["label"]
groups = clean_df["hotel"]

# Create hotel-grouped 5-fold splits
cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)
folds = list(cv.split(X_text, y, groups=groups))

# Check hotel leakage
fold_summary = []
fold_rows = []
for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
    train_hotels = set(groups.iloc[train_idx])
    val_hotels = set(groups.iloc[val_idx])
    assert train_hotels.isdisjoint(val_hotels)
    fold_summary.append({
        "fold": fold_id,
        "train_reviews": len(train_idx),
        "val_reviews": len(val_idx),
        "train_hotels": len(train_hotels),
        "val_hotels": len(val_hotels),
        "train_deceptive_rate": round(y.iloc[train_idx].mean(), 3),
        "val_deceptive_rate": round(y.iloc[val_idx].mean(), 3),
    })
    fold_rows.extend(
        {"row_index": int(idx), "fold": fold_id}
        for idx in val_idx
    )

summary_df = pd.DataFrame(fold_summary)
print(summary_df)

# Save fold assignments
fold_assignment = pd.DataFrame(fold_rows).sort_values("row_index")
fold_assignment.to_csv(OUTPUT_DIR / "fold_assignments.csv", index=False)

# Step 6: TF-IDF
def get_fold_features(clean_df, train_idx, val_idx):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), # Extract unigrams and bigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    # Fit TF-IDF on training data
    X_train = vectorizer.fit_transform(clean_df.iloc[train_idx]["review_text"])
    X_val = vectorizer.transform(clean_df.iloc[val_idx]["review_text"])

    # Extract labels
    y_train = clean_df.iloc[train_idx]["label"].to_numpy()
    y_val = clean_df.iloc[val_idx]["label"].to_numpy()

    return X_train, X_val, y_train, y_val, vectorizer


# Generate TF-IDF features for all five folds
fold_features = []

for train_idx, val_idx in folds:
    X_train, X_val, y_train, y_val, vectorizer = get_fold_features(
        clean_df, train_idx, val_idx
    )

    fold_features.append((X_train, X_val, y_train, y_val))

# ### Preprocessing Summary
#
# **Files provided:**
# - `cleaned_reviews.csv`: Cleaned review texts and encoded labels (0 = truthful, 1 = deceptive).
# - `fold_assignments.csv`: Predefined hotel-grouped 5-fold cross-validation splits.
#
# **Preprocessing completed:**
# - Data cleaning and label encoding.
# - Hotel-grouped 5-fold cross-validation.
# - TF-IDF feature extraction for all five folds.

