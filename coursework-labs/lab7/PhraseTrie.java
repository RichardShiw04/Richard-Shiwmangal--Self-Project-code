package lab7;
import java.util.ArrayList;
import java.util.List;

public class PhraseTrie<VT> extends AbstTrie<List<String>, String, VT> {
    @Override
    protected int key_length(List<String> key) {
        return key.size();
    }

    @Override
    protected String key_component_at(List<String> key, int index) {
        return key.get(index);
    }

    // Non-destructive add operation for a new phrase component
    public List<String> add_component(List<String> key, String component) {
        List<String> newKey = new ArrayList<>(key);
        newKey.add(component);
        return newKey;
    }
}
