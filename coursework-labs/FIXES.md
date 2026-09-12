# Fixes and verification

## Source and scope

Copied from local `Documents/School Fall Semester 2024/{lab8,lab7,astar,maze lab 2}` and `Downloads/DeckCards`. Originals were not edited. The Downloads maze variant differed only in colors; the Documents version was selected. The user's clarification requested all three game candidates, so Word Finder, A* and DeckCards are included. Unrelated coursework was excluded. Supplied images, saved maps, dictionaries, DNA samples, instructor code and attribution remain with their projects.

Build products, local environment setup (`.envrc`) and redundant compiled classes are excluded. The original project directory/package names are retained under a new `coursework-labs/` directory in the repository; existing repository files are unchanged.

## Edited source files

- `lab8/NeedlemanWunsch.java`: prevent out-of-bounds traceback when sequence B is exhausted; apply terminal deletion penalties only at the last column and insertion penalties only at the last row; return an empty Optional for an empty DNA file; guard an empty matrix in the print helper; add algorithm/traceback Javadocs. Scoring constants, embedded demo and data are preserved.
- `maze lab 2/maze.java`: stop DFS backtracking at an exhausted root instead of looping forever; guard absent traceback paths; return on traceback failure instead of displaying a misleading start message; remove the redundant trace call because the base invokes it; add class documentation. The supplied mazebase is untouched.
- `lab7/WordFinder.java`: remove blocking console search; use the GUI word; preserve the result as state drawn during repaint; initialize before visibility; start on Swing's event thread; normalize case and strip numeric dictionary ranks (including `4a`); clear results when typing or clearing.
- `lab7/StringTrie.java`: include terminal nodes in completion streams; reset continuation on clear; represent failed prefixes as no current node and return an empty stream.
- `astar/myastar.java`: remove the incomplete duplicate class declaration and duplicated template preamble; retain the completed implementation, package and original remaining help comments; guard input coordinates; permit zero-cost terrain and use a zero heuristic in that case. Positive costs retain hex-distance A*. No other A* Java sources were changed.
- `DeckCards/Deck.cpp`: guard initialization ranges/capacity before writes; make shuffle a no-op for fewer than two cards; replace a multicharacter character literal with a string for readable card indices.
- `DeckCards/card_game.cpp`: print the returned pile-card text; include time/input-limit headers; validate numeric input and indices, exit on EOF, and enforce matching suit/face while retaining the current turn on rejection.
- `DeckCards/makefile`: explicitly select C++11 and compiler warnings. Copied source timestamps were normalized locally to avoid warnings from originals dated 2107.

Added top-level and per-project READMEs, this fixes report, `.gitignore`, and regression tests with `tests/run.sh`.

## Validation

- All Java projects compile with JDK 21; real maze Swing sources compile independently of its test stub.
- 108 sequence/scheme combinations: equal alignment lengths, original sequence reconstruction, rescored traceback equals computed score, including empty inputs. Empty file and empty print matrix pass. Original large DNA main completed.
- 30 random mazes: valid adjacent parent chain to the start; unreachable exit terminates. This uses a test-only drawing stub, not a graphical end-to-end test.
- Trie lookup/prefix inclusion, nonexistent prefix, clear/reset checks pass.
- 100 deterministic random A* maps with blocked, weighted and zero-cost cells match an independent repeated-relaxation shortest-path computation. Invalid source coordinates return empty.
- Card/deck compile with C++11 and warnings; original test_deck demonstration runs; added assertions cover empty/single shuffle, copy independence and oversized initialization. Undefined-behavior sanitizer regression run passed. AddressSanitizer could not initialize in this host environment, so no AddressSanitizer result is claimed.
- Card game smoke input covers malformed input, invalid index, drawing and EOF.

Graphical rendering and a complete interactive game session have not been manually verified. The supplied maze/pathfinder drawing framework is retained. No assignment brief beyond the local code/comments and A* README was available; fixes preserve the recovered assignment algorithms and rules.
