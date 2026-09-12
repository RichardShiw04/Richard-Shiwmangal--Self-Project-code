# Project updates

- Replaced executable CSV genre parsing with `ast.literal_eval` in the transformer script.
- Bounded the transformer sample size by the available training rows; datasets still need enough rows for the train/validation split.
- Added `MOVIE_DATA_DIR` to both scripts while retaining Downloads as the fallback.
- Corrected an inaccurate NLTK-download status message in the baseline script.
- Added dependencies and a project-focused README. Original assignment notes are preserved separately.

Both scripts pass Python parsing. Training and reported model metrics were not rerun: the dataset/model downloads and training are outside this upload verification. The baseline source retains its original manual probability calculations and scikit-learn classifier sections. No claims of full assignment completion or generalization accuracy are made.
