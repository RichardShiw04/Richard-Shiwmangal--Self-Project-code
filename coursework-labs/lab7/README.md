# Word Finder — lab7

Swing word lookup backed by `StringTrie`, plus the original WordTrie, AbstTrie and PhraseTrie exercises.

From the coursework-labs directory:

```sh
mkdir -p build/words
javac -d build/words lab7/*.java
(cd lab7 && java -cp ../build/words lab7.WordFinder)
```

Run from lab7 so `hfwords.txt` is found. Type letters, press Enter to search, and Space to clear. Dictionary loading strips leading numeric ranks and normalizes case. A missing dictionary prints an error to the terminal.

Fixes: search the word typed in the window without blocking for console input; retain results during repaint; initialize before showing the window and launch on Swing's event thread; parse numbered dictionary entries; return trie terminal entries; correctly clear/reset continuation state and return no completions for missing prefixes.
