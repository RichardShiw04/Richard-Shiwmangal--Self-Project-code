package lab7;
import java.util.*;

public abstract class AbstTrie<KT, KCT, VT> {
    protected class Node {
        Map<KCT, Node> children = new HashMap<>();
        SVPair value; // Optional value stored at this node
    }

    protected Node root = new Node();

    protected abstract int key_length(KT key);

    protected abstract KCT key_component_at(KT key, int index);

    public void insert(KT key, VT value) {
        Node current = root;
        for (int i = 0; i < key_length(key); i++) {
            KCT component = key_component_at(key, i);
            current.children.putIfAbsent(component, new Node());
            current = current.children.get(component);
        }
        current.value = new SVPair(value);
    }

    public Optional<VT> search(KT key) {
        Node current = root;
        for (int i = 0; i < key_length(key); i++) {
            KCT component = key_component_at(key, i);
            current = current.children.get(component);
            if (current == null) return Optional.empty();
        }
        return current.value != null ? Optional.of((VT) current.value.value) : Optional.empty();
    }

    protected class SVPair {
        VT value;

        public SVPair(VT value) {
            this.value = value;
        }
    }
}
