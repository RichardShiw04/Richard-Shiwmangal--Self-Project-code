

import pandas as pd
import os

# Define the path to the CSV files
csv_path = os.path.expanduser('~/Downloads')
train_df = pd.read_csv(os.path.join(csv_path, 'train.csv'))
test_df = pd.read_csv(os.path.join(csv_path, 'test.csv'))
genres_df = pd.read_csv(os.path.join(csv_path, 'movies_genres.csv'))

"""Naive bayes

Inspect Data and Identify Columns
 understand the structure of `test_df`
"""

print("Test DataFrame Head:")
print(test_df.head())
print("\nTest DataFrame Info:")
test_df.info()

"""
To fully address the subtask of examining both `train_df` and `test_df`, I will now inspect the `train_df` by displaying its first 5 rows and its column information.


"""

print("Train DataFrame Head:")
print(train_df.head())
print("\nTrain DataFrame Info:")
train_df.info()

""" Text Preprocessing


Create a Python function to preprocess text data. This function will handle tokenization, lowercasing, removal of special characters, and stop-word removal. For 'terms/NERs', the initial approach will be to include all significant terms after standard cleaning, but specific NER handling can be incorporated if needed later.

"""

import re

# Common English stop words (without NLTK dependency)
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'what', 'which', 'who',
    'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'can', 'just', 'should', 'now', 'is', 'am', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'would', 'could', 'ought', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
    'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'of', 'in', 'to', 'for',
    'by', 'from', 'as', 'at', 'up', 'with', 'about', 'into', 'through', 'during'
}

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and punctuation, keep only lowercase letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # Simple word tokenization by splitting on whitespace
    tokens = text.split()

    # Remove stop words
    filtered_tokens = [word for word in tokens if word and word not in STOP_WORDS]

    return filtered_tokens

print("NLTK data (punkt, stopwords, and punkt_tab) downloaded and preprocessing function 'preprocess_text' defined.")

# Apply the preprocessing function to the 'overview' column in both dataframes

train_df['processed_overview'] = train_df['overview'].apply(preprocess_text)
test_df['processed_overview'] = test_df['overview'].apply(preprocess_text)

print("Preprocessed 'overview' columns added to train_df and test_df.")
print("\nTrain DataFrame with processed_overview head:")
print(train_df[['overview', 'processed_overview']].head())
print("\nTest DataFrame with processed_overview head:")
print(test_df[['overview', 'processed_overview']].head())

"""
Apply the preprocessing function to the training text data. From the preprocessed text, build a unique vocabulary of words. Then, convert the text into numerical features using a Bag-of-Words (BoW) representation, counting the occurrences of each word in the vocabulary for each document.
"""

from sklearn.feature_extraction.text import CountVectorizer
import ast

# Join the processed tokens back into a string for CountVectorizer
train_df['processed_overview_str'] = train_df['processed_overview'].apply(lambda x: ' '.join(x))
test_df['processed_overview_str'] = test_df['processed_overview'].apply(lambda x: ' '.join(x))

# 1. Initialize a CountVectorizer
# The tokenizer is set to lambda x: x.split(' ') because the text is already preprocessed into lists of words,
# which are then joined into a string before being passed to the vectorizer.
# This prevents the CountVectorizer from performing its own tokenization, which might not align with NLTK's.
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '))

# 2. Fit the CountVectorizer to the training data and 3. Transform the training data
X_train_bow = vectorizer.fit_transform(train_df['processed_overview_str'])

# 4. Transform the test data using the same fitted CountVectorizer
X_test_bow = vectorizer.transform(test_df['processed_overview_str'])

# 5. Extract and convert genre labels from genre_ids column
y_train_genres = train_df['genre_ids'].apply(ast.literal_eval).tolist()

# 6. Print the shape of X_train_bow, X_test_bow, and the first few entries of y_train_genres
print(f"Shape of X_train_bow: {X_train_bow.shape}")
print(f"Shape of X_test_bow: {X_test_bow.shape}")
print("First 5 entries of y_train_genres:")
print(y_train_genres[:5])



from sklearn.preprocessing import MultiLabelBinarizer

# 1. Initialize a MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# 2. Fit the MultiLabelBinarizer to y_train_genres to discover all unique genre labels
# 3. Transform y_train_genres into a binary (multi-hot encoded) matrix
y_train_binarized = mlb.fit_transform(y_train_genres)

# 4. Print the shape of y_train_binarized and the first few rows to verify the transformation
print(f"Shape of y_train_binarized: {y_train_binarized.shape}")
print("First 5 entries of y_train_binarized:")
print(y_train_binarized[:5])
print("\nClasses (genre IDs) found by MultiLabelBinarizer:")
print(mlb.classes_)

""" Calculate  Prior Probabilities

Compute the prior probability P(Class) for each genre (class) based on the frequency of each class in the training dataset.

To compute the prior probability P(Class) for each genre, I will sum the occurrences for each genre from the `y_train_binarized` array, divide by the total number of training samples, and store these probabilities in a pandas Series mapped to their respective genre IDs, as instructed.
"""

import numpy as np

# 1. Calculate the sum of occurrences for each genre
genre_counts = np.sum(y_train_binarized, axis=0)

# 2. Determine the total number of training samples
total_samples = y_train_binarized.shape[0]

# 3. Divide the sum of occurrences by the total number of training samples to get prior probabilities
prior_probabilities = genre_counts / total_samples

# 4. Store these prior probabilities, mapping genre IDs to their calculated probabilities
prior_prob_series = pd.Series(prior_probabilities, index=mlb.classes_)

# 5. Print the calculated prior probabilities for each genre
print("Prior Probabilities for each genre:")
print(prior_prob_series)

""" Calculate Conditional Probabilities 


For each word in the vocabulary and each class, calculate the conditional probability P(Word | Class). Implement Laplace smoothing (add-one smoothing) to handle words that may not appear in certain classes, preventing zero probabilities.

"""

import numpy as np

# 1. Initialize an empty dictionary to store conditional probabilities
conditional_probabilities = {}

# 2. Get the total number of unique words in the vocabulary
vocabulary_size = len(vectorizer.vocabulary_)

# Get a mapping from feature index to word for easier inspection
feature_names = vectorizer.get_feature_names_out()

# 3. Iterate through each genre (class)
for genre_index, genre_id in enumerate(mlb.classes_):
    # 4. Identify the indices of the training samples that belong to this genre
    # y_train_binarized is (num_samples, num_genres), so select rows where the current genre column is 1
    genre_samples_indices = np.where(y_train_binarized[:, genre_index] == 1)[0]

    # Ensure there are samples for this genre
    if len(genre_samples_indices) == 0:
        # If no samples for this genre, probabilities for this genre will be uniform smoothed probabilities
        # (1) / (vocabulary_size) for all words, given the total words in genre is 0.
        # This handles the case where a genre exists in mlb.classes_ but has no corresponding documents
        # in the current training set (though unlikely for this dataset).
        total_words_in_genre = 0 # No actual words from documents in this genre
        word_counts_in_genre = np.zeros(vocabulary_size) # No words observed
    else:
        # 5. Sum the word counts from X_train_bow for all documents belonging to the current genre
        # X_train_bow[genre_samples_indices] selects the rows (documents) that belong to this genre
        # .sum(axis=0) sums the word counts across these documents
        # .A converts sparse matrix row to a dense numpy array
        word_counts_in_genre = np.array(X_train_bow[genre_samples_indices].sum(axis=0)).flatten()

        # 6. Calculate the total number of words in documents belonging to the current genre
        total_words_in_genre = np.sum(word_counts_in_genre)

    # Initialize dictionary for current genre's conditional probabilities
    conditional_probabilities[genre_id] = {}

    # 7. For each word in the vocabulary, calculate the conditional probability P(Word | Class) with Laplace smoothing
    for word_idx in range(vocabulary_size):
        # P(word | class) = (word_count_in_genre + 1) / (total_words_in_genre + vocabulary_size)
        p_word_given_genre = (word_counts_in_genre[word_idx] + 1) / (total_words_in_genre + vocabulary_size)
        # 8. Store these probabilities
        conditional_probabilities[genre_id][feature_names[word_idx]] = p_word_given_genre

# 9. Optionally, print a small sample of the calculated conditional probabilities
print("Sample Conditional Probabilities (P(Word | Class)):")
# Choose a few example genres and words
example_genres = list(mlb.classes_[:3]) # First 3 genres
example_words = [feature_names[0], feature_names[1], feature_names[100]] # First two words and one further word

for genre_id in example_genres:
    print(f"\nGenre ID: {genre_id}")
    for word in example_words:
        if word in conditional_probabilities[genre_id]:
            print(f"  P('{word}' | Class {genre_id}): {conditional_probabilities[genre_id][word]:.6f}")
        else:
            print(f"  Word '{word}' not found in vocabulary for Genre {genre_id}")

print(f"Conditional probabilities calculated for {len(mlb.classes_)} genres and {vocabulary_size} words.")

"""  Naive Bayes Classifier Prediction using OneVsRestClassifier


Use scikit-learn's OneVsRestClassifier with MultinomialNB for proper multi-label classification.

OneVsRestClassifier properly handles the multi-label nature of the problem by training independent

binary classifiers for each genre. 
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier

# 1. Train OneVsRestClassifier with MultinomialNB
print("Training OneVsRestClassifier with MultinomialNB...")
ovr_classifier = MultiOutputClassifier(MultinomialNB())
ovr_classifier.fit(X_train_bow, y_train_binarized)

print("OneVsRestClassifier training complete!")

# 2. Make predictions on training data for evaluation
y_train_predicted_binarized_ovr = ovr_classifier.predict(X_train_bow)

print("Shape of y_train_predicted_binarized (OneVsRest):", y_train_predicted_binarized_ovr.shape)
print("First 5 entries of y_train_predicted_binarized (OneVsRest):")
print(y_train_predicted_binarized_ovr[:5])

# 3. Make predictions on test data
y_test_predicted_binarized_ovr = ovr_classifier.predict(X_test_bow)

print("Shape of y_test_predicted_binarized (OneVsRest):", y_test_predicted_binarized_ovr.shape)
print("First 5 entries of y_test_predicted_binarized (OneVsRest):")
print(y_test_predicted_binarized_ovr[:5])

# Convert predictions to genre ID format for submission
test_predictions = []
for idx in range(len(y_test_predicted_binarized_ovr)):
    # Get all genres predicted for this sample (where value is 1)
    predicted_genre_indices = np.where(y_test_predicted_binarized_ovr[idx] == 1)[0]
    predicted_genres = [mlb.classes_[i] for i in predicted_genre_indices]
    
    # If no genres predicted, use the most common genre as fallback
    if not predicted_genres:
        predicted_genres = [mlb.classes_[np.argmax(y_train_binarized.sum(axis=0))]]
    
    # Convert list of genre IDs to space-separated string
    predicted_genres_str = ' '.join(map(str, sorted(predicted_genres)))
    test_predictions.append(predicted_genres_str)

# Add predictions to the test_df
test_df['predicted_genre_ids'] = test_predictions

print("Predictions for all test documents generated using OneVsRestClassifier.")
print("First 5 test predictions:")
print(test_df[['movie_id', 'title', 'predicted_genre_ids']].head())

""" Predict and Evaluate on Training Data

The OneVsRestClassifier provides multi-label predictions directly,  evaluate
against the true multi-label targets without needing to convert single-label predictions.
"""

from sklearn.metrics import classification_report

print("\nClassification Report for Training Data (OneVsRestClassifier):\n")
target_names = [str(gid) for gid in mlb.classes_]
print(classification_report(y_train_binarized, y_train_predicted_binarized_ovr, 
                           target_names=target_names, zero_division=0))

# Calculate exact match accuracy (all genres must match exactly)
exact_matches = (y_train_binarized == y_train_predicted_binarized_ovr).all(axis=1).sum()
exact_match_accuracy = exact_matches / len(y_train_binarized)
print(f"\nExact Match Accuracy (all genres correct): {exact_match_accuracy:.4f}")

# Calculate hamming loss (proportion of incorrect labels)
from sklearn.metrics import hamming_loss
train_hamming = hamming_loss(y_train_binarized, y_train_predicted_binarized_ovr)
print(f"Hamming Loss (lower is better): {train_hamming:.4f}")


print("\nTest Set Predictions Complete")
print(f"Shape of y_test_predicted_binarized (OneVsRest): {y_test_predicted_binarized_ovr.shape}")
print("First 5 entries of y_test_predicted_binarized (OneVsRest):")
print(y_test_predicted_binarized_ovr[:5])

print("\nNote: True labels are not available for the test set, so quantitative evaluation metrics cannot be computed.")



from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt

# 1. Get classification report as dictionary
report_dict = classification_report(
    y_train_binarized,
    y_train_predicted_binarized_ovr,
    target_names=[str(gid) for gid in mlb.classes_],
    zero_division=0,
    output_dict=True
)

# Extract F1-scores for each class (genre ID)
f1_scores = {}
for genre_id_str in mlb.classes_:
    if str(genre_id_str) in report_dict and 'f1-score' in report_dict[str(genre_id_str)]:
        f1_scores[genre_id_str] = report_dict[str(genre_id_str)]['f1-score']

# 2. Create a Pandas DataFrame from the extracted F1-scores
f1_df = pd.DataFrame(f1_scores.items(), columns=['genre_id', 'f1_score'])

# Map genre IDs to their corresponding genre names using genres_df
f1_df['genre_id'] = f1_df['genre_id'].astype(int)
genres_df['id'] = genres_df['id'].astype(int)
f1_df = pd.merge(f1_df, genres_df, left_on='genre_id', right_on='id', how='left')

# Sort by F1-score for better visualization
f1_df = f1_df.sort_values(by='f1_score', ascending=False)

# 3. Create a bar chart of these F1-scores
plt.figure(figsize=(12, 7))
plt.bar(f1_df['name'], f1_df['f1_score'], color='skyblue')

# 4. Label the axes
plt.xlabel('Genre')
plt.ylabel('F1-Score')

# 5. Add a title to the plot
plt.title('F1-Scores per Genre on Training Data (OneVsRestClassifier)')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add values on top of bars
for index, row in f1_df.iterrows():
    plt.text(index, row['f1_score'], f"{row['f1_score']:.2f}", ha='center', va='bottom', fontsize=8)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(os.path.join(csv_path, 'f1_scores_chart.png'), dpi=100, bbox_inches='tight')
plt.close()

print("Bar chart of F1-Scores per Genre on Training Data saved as 'f1_scores_chart.png'.")

"""## Generate Submission File

### Subtask:
Create a `submission.csv` file with movie IDs and multi-label genre predictions from the OneVsRestClassifier.
"""

# 1. Create a new DataFrame for the submission file
submission_df = pd.DataFrame({
    'movie_id': test_df['movie_id'],
    'genre_ids': test_df['predicted_genre_ids']
})

# 2. Save this new DataFrame to a CSV file named submission.csv without including the DataFrame index
output_path = os.path.join(csv_path, 'submission.csv')
submission_df.to_csv(output_path, index=False)

print(f"Submission file 'submission.csv' created successfully at {output_path}")
print("First 5 rows of the submission file:")
print(submission_df.head())

# 3. Verify the submission format matches sample_submission.csv
print("\nSubmission File Validation:")
print(f"Total predictions: {len(submission_df)}")
print(f"Columns: {submission_df.columns.tolist()}")

# Verify all movie_ids are in test set
test_movie_ids = set(test_df['movie_id'].astype(int).tolist())
submission_movie_ids = set(submission_df['movie_id'].astype(int).tolist())
if submission_movie_ids == test_movie_ids:
    print(f"✓ All {len(submission_movie_ids)} movie_ids match the test set")
else:
    print("✗ Some movie_ids do not match the test set")

# Verify all predicted genres are valid
genres_df_local = pd.read_csv(os.path.join(csv_path, 'movies_genres.csv'))
valid_genre_ids = set(genres_df_local['id'].astype(int).tolist())

# Extract unique genre IDs from submission
all_genres_in_submission = set()
for genre_str in submission_df['genre_ids']:
    genre_ids = [int(x) for x in str(genre_str).split()]
    all_genres_in_submission.update(genre_ids)

if all_genres_in_submission.issubset(valid_genre_ids):
    print(f"✓ All {len(all_genres_in_submission)} unique genre IDs are valid")
    print(f"✓ Genre IDs used: {sorted(all_genres_in_submission)}")
else:
    invalid = all_genres_in_submission - valid_genre_ids
    print(f"✗ Invalid genre IDs found: {sorted(invalid)}")

print("✓ Submission file is ready for submission!")