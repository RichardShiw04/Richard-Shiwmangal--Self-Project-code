# Coursework labs

Recovered Java and C++ coursework, with original directory and package names retained. Requires **JDK 17+**, and **C++11 or newer plus make** for DeckCards. Verified with JDK 21 and Apple clang on macOS.

| Project | Description | Instructions |
| --- | --- | --- |
| lab8 | Needleman–Wunsch DNA alignment, three scoring schemes | [README](lab8/README.md) |
| maze lab 2 | Recursive maze generation, depth-first solve and traceback | [README](maze%20lab%202/README.md) |
| lab7 | Swing Word Finder and trie exercises | [README](lab7/README.md) |
| astar | Hex-grid A* pathfinding with graphical assets and saved maps | [README](astar/README.md) |
| DeckCards | Two-player UNO-like card game against a computer | [README](DeckCards/README.md) |

Run `sh tests/run.sh` from this directory for repeatable regression checks. GUI applications require a desktop display; the maze algorithm test replaces drawing with a test-only stub and does not verify its graphical rendering.

See [FIXES.md](FIXES.md) for exact edits, scope, provenance, and validation limitations. Instructor comments and original acknowledgements are retained; no new license is asserted for supplied coursework or assets.
