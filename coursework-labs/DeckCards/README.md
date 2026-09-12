# DeckCards — UNO-like card game

Standard-card matching game: the human and computer start with seven cards each. Play a card matching the pile's suit or face, or draw with -1. The first empty hand wins; the original rule also ends play when the draw deck is exhausted. The Card model retains its original allowed faces 1–14, while this game deals 1–13.

```sh
cd DeckCards
make
./card_game
./test_deck
make clean
```

Requires a C++11 compiler and make. Type the displayed zero-based index or -1. EOF exits cleanly; invalid input, out-of-range indices and nonmatching cards keep the human turn. A rejected nonmatching card returns to the top of the hand, so read the refreshed indices.

Fixes: show the pile card and readable indices; validate input and matching rules; avoid empty/single-card shuffle failures; reject initialization beyond deck capacity; include required standard headers and explicit compiler standard. Existing demonstration tests are retained; additional assertions live in ../tests.
