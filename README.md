# Movie Genre Classification

Predicting movie genres from plot summaries with a bag-of-words baseline and a fine-tuned DistilBERT model.

## Implementations

- `self_project_richard_shiwmangal.py`: text preprocessing, manual Naive Bayes probability calculations, scikit-learn classification/evaluation sections, and a submission export.
- `bert_genre_fast.py`: a bounded training sample, a 70/30 training/validation split, DistilBERT fine-tuning, and prediction export.

## Run

Requires Python, the dependencies below, and the original movie-genre CSV dataset. No dataset or trained model is bundled.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export MOVIE_DATA_DIR=/path/to/movie-data
python self_project_richard_shiwmangal.py
python bert_genre_fast.py
```

The data directory must contain `train.csv`, `test.csv`, and `movies_genres.csv`. It defaults to `~/Downloads`. The scripts retain their original output paths; check them before running to avoid replacing an existing submission. Transformer execution downloads pretrained model files and may take substantial time on a CPU.

Both scripts pass Python parsing. Model training was not rerun, and no new accuracy results are claimed. See [PROJECT-UPDATES.md](PROJECT-UPDATES.md) for targeted fixes and [ORIGINAL-NOTES.md](ORIGINAL-NOTES.md) for preserved source requirements. Existing [Acknowledgements.md](Acknowledgements.md) is retained.

## Other projects

The Java, C++, and database projects now live in separate repositories:

| Project | Description |
| --- | --- |
| [Anime Vector Search](https://github.com/RichardShiw04/anime-vector-search) | A notebook comparing semantic retrieval with a simple lexical TF-IDF baseline over anime descriptions. Sentence Transformers generates MiniLM embeddings; Qdrant stores them in memory and returns similar descriptions. |
| [Card Matching Game](https://github.com/RichardShiw04/card-matching-game) | A C++ card-matching game against a computer, using custom Card and Deck classes. |
| [DNA Sequence Alignment](https://github.com/RichardShiw04/dna-sequence-alignment) | Needleman–Wunsch sequence alignment with three scoring schemes, traceback reconstruction, and DNA sample inputs. |
| [Hex Grid Pathfinding](https://github.com/RichardShiw04/hex-grid-pathfinding) | A* search across a weighted hexagonal map, with a hashed-heap frontier and animated path visualization. |
| [Jam Ingredients Database](https://github.com/RichardShiw04/jam-ingredients-database) | A PostgreSQL relational model for jams, ingredients, and prices. Two foreign keys link each jam to its ingredients, and a join produces a readable product report. |
| [Maze Pathfinder](https://github.com/RichardShiw04/maze-pathfinder) | Random maze generation, depth-first pathfinding, and parent-path visualization in Java. |
| [OrientDB Order Manager](https://github.com/RichardShiw04/orientdb-order-manager) | A Node.js store simulation with product seeding, random orders, customer invoices, price updates, and a recurring sales report. Orders retain item-price snapshots. |
| [PostgreSQL Query Tools](https://github.com/RichardShiw04/postgresql-query-tools) | Small Python utilities for connecting to PostgreSQL and printing query results. |
| [Stock Market Dashboard](https://github.com/RichardShiw04/stock-market-dashboard) | A React dashboard and FastAPI backend for stock quotes, charts, fundamentals, news, and user profiles. Finnhub provides market data; Supabase provides authentication and PostgreSQL-backed profile/watchlist storage. |
| [Student Grades Database](https://github.com/RichardShiw04/student-grades-database) | PostgreSQL tables and joins for students, courses, and final grades. |
| [Word Finder](https://github.com/RichardShiw04/word-finder) | A Java desktop word-search application backed by a trie, with prefix traversal exercises. |

See [my profile](https://github.com/RichardShiw04) for Inbox Lens and the full project index.
