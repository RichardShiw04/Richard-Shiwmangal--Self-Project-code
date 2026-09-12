# A* hex-grid pathfinding

The recovered `myastar` uses the supplied HashedHeap frontier, visited grid and parent-linked coordinates. Positive terrain costs use hex distance as the heuristic; zero-cost terrain uses a zero heuristic. Negative-cost terrain is impassable. Original default customization keeps all terrain costs at 1.

From the coursework-labs directory:

```sh
mkdir -p build/astar
javac -d build/astar astar/*.java
(cd astar && java -cp ../build/astar astar.myastar)
# Replay a supplied map:
(cd astar && java -cp ../build/astar astar.myastar config0.run)
```

Requires a desktop display. Run inside astar to resolve images and map files. The application writes `myastar.run` when generating a map; copy it first if you want to retain that sample. Original [assignment instructions](ASTARREADME.md) describe additional arguments; use the package-qualified commands above for this source layout. No instructor-server download/upload was performed.

Fixes: remove the accidental duplicate class header/import that prevented compilation, reject out-of-range coordinates, and support zero-cost paths without an overestimating heuristic. Framework files and assets remain unchanged.
