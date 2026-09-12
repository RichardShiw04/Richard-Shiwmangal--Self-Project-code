# Maze generation and solving

This recovered project combines maze generation and solver/trace portions of the assignments. `maze` extends the supplied `mazebase` Swing framework. It carves a random 41×41 maze using recursive depth-first search, explores passages from (row 1, column 1), then traces parents from the right-edge exit. The generated maze is a tree, so its unique solution is also shortest; this is not a general shortest-path solver for cyclic mazes.

From the coursework-labs directory:

```sh
mkdir -p build/maze
javac -d build/maze "maze lab 2/"*.java
java -cp build/maze maze
```

Requires a desktop display. Generation and solving start automatically. Close the window to exit. This version does not implement interactive play; the base class keyboard handler only prints key codes. Dimensions/colors can be adjusted in `customize()`; keep odd dimensions.

Fixes: stop when an unreachable exit exhausts the search, skip tracing a missing path, and trace once through the base lifecycle. The instructor base is unchanged. `Downloads/maze-1.java` was a duplicate differing only in wall/dot colors, so it is not compiled as a second conflicting `maze` class.
