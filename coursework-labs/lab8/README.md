# DNA alignment — lab8

Needleman–Wunsch dynamic programming stores score and traceback matrices in O(mn) time and space. Scheme1 awards +1 for exact matches and 0 otherwise; Scheme2 uses +1 matches, -1 mismatches and gaps; Scheme3 uses case-insensitive +3 matches, -1 mismatches, -2 internal gaps and free terminal gaps. Ties prefer diagonal, then deletion, then insertion.

From the coursework-labs directory:

```sh
mkdir -p build/dna
javac -d build/dna lab8/*.java
java -cp build/dna lab8.lab8main
```

The original main prints a score and two aligned strings for its embedded Scheme3 example. Four original `.dna` sample files are included. `load_dna` reads their first line and returns an empty Optional for an empty or unreadable file; these are assignment inputs, not a biological identification tool. To use other sequences, instantiate a scheme from code in package lab8, call `computeAlignment()`, then `traceback()`.

Fixes: handle exhausted-sequence traceback safely, apply terminal gaps to the appropriate sequence, and handle empty file/matrix input. Original main and sample sequences are unchanged.
