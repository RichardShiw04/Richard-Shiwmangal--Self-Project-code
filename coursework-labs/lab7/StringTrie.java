package lab7;
import java.util.Map;
import java.util.HashMap;
import java.util.stream.Stream;
import java.util.AbstractMap;

// Define the TrieNode class for managing nodes in the Trie
class TrieNode<V> {
    Map<Character, TrieNode<V>> children = new HashMap<>();
    V value = null;

    public TrieNode<V> getChild(char c) {
        return children.get(c);
    }

    public void addChild(char c, TrieNode<V> node) {
        children.put(c, node);
    }

    public Stream<Map.Entry<String, V>> getChildrenStream(String prefix) {
        Stream<Map.Entry<String, V>> own = value == null ? Stream.empty()
                : Stream.of(new AbstractMap.SimpleEntry<>(prefix, value));
        return Stream.concat(own, children.entrySet().stream()
                .flatMap(entry -> entry.getValue().getChildrenStream(prefix + entry.getKey())));
    }
}

// StringTrie class
public class StringTrie<V> {




    private TrieNode<V> root = new TrieNode<>();
    private TrieNode<V> currentNode;
    private StringBuilder currentPrefix = new StringBuilder();

    public StringTrie() {
        this.currentNode = root; // Start at the root node
    }

    // Add a key-value pair to the trie
    public void add(String key, V value) {
        TrieNode<V> current = root;
        for (char ch : key.toCharArray()) {
            current.children.putIfAbsent(ch, new TrieNode<>());
            current = current.children.get(ch);
        }
        current.value = value;
    }
    public void clear() {
        root = new TrieNode<>();
        reset_continuation();
    }

    
    // Get the value for a given key
    public V get(String key) {
        TrieNode<V> current = root;
        for (char ch : key.toCharArray()) {
            current = current.children.get(ch);
            if (current == null) return null;
        }
        return current.value;
    }
    

    // Get a stream of entries under the current prefix
    public Stream<Map.Entry<String, V>> current_stream(int limit) {
        return currentNode == null ? Stream.empty() : currentNode.getChildrenStream(currentPrefix.toString()).limit(limit);
    }

    // Continue search with the next character
    public void continue_search(char c) {
        currentPrefix.append(c);
        if (currentNode != null) currentNode = currentNode.getChild(c);
    }

    // Initialize continuation with a prefix
    public void begin_continuation(String prefix) {
        currentPrefix.setLength(0);  // Clear the current prefix
        currentPrefix.append(prefix);
        currentNode = root;
        for (char c : prefix.toCharArray()) {
            TrieNode<V> nextNode = currentNode.getChild(c);
            if (nextNode == null) {
                currentNode = null;
                return; // Missing prefix has no completions.
            }
            currentNode = nextNode;
        }
    }

    // Reset the search continuation
    public void reset_continuation() {
        currentNode = root;
        currentPrefix.setLength(0); // Reset the prefix
    }

    // Get the current key during a search
    public String current_key() {
        return currentPrefix.toString();
    }
}
