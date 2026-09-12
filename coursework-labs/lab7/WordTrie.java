package lab7;
import java.util.Map;
import java.util.HashMap;
import java.util.Optional;

public class WordTrie {
    private Map<String, Integer> wordMap;

    public WordTrie() {
        wordMap = new HashMap<>();
    }

    // This method should return Optional<Integer> instead of Integer directly
    public Optional<Integer> get(String key) {
        Integer rank = wordMap.get(key);  // Get the rank (could be null if not found)
        return Optional.ofNullable(rank);  // Return it wrapped in Optional
    }

    // This method sets the rank for a word
    public void set(String key, Integer rank) {
        wordMap.put(key, rank);  // Store the word with its rank
    }

    // Other methods can be added to interact with the trie as needed
}
