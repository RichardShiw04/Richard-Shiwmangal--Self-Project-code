package lab7;
public class TrieTest {
 public static void main(String[] args) {
  StringTrie<Integer> t=new StringTrie<>();t.add("a",1);t.add("ant",2);t.add("bat",3);
  t.begin_continuation("a");if(t.current_stream(10).count()!=2)throw new AssertionError();
  t.continue_search('x');if(t.current_stream(10).count()!=0)throw new AssertionError();
  t.begin_continuation("bad");if(t.current_stream(10).count()!=0)throw new AssertionError();
  t.clear();if(t.get("a")!=null || t.current_stream(10).count()!=0)throw new AssertionError();
  t.add("new",4);t.begin_continuation("new");if(t.current_stream(1).count()!=1)throw new AssertionError();
  System.out.println("Trie lookup, prefix, missing-prefix and clear checks passed.");
 }
}
