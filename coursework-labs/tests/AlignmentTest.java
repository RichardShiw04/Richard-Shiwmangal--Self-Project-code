package lab8;
import java.io.*;
public class AlignmentTest {
 static void check(boolean ok) { if (!ok) throw new AssertionError(); }
 public static void main(String[] args) throws Exception {
  String[] inputs={"", "A", "T", "AC", "CAA", "GATTACA"};
  int count=0;
  for(String a:inputs) for(String b:inputs) for(int scheme=1;scheme<=3;scheme++) {
   NeedlemanWunsch n=scheme==1?new Scheme1(a,b):scheme==2?new Scheme2(a,b):new Scheme3(a,b);
   int score=n.computeAlignment();
   ByteArrayOutputStream bytes=new ByteArrayOutputStream(); PrintStream old=System.out;
   try { System.setOut(new PrintStream(bytes)); n.traceback(); } finally {System.setOut(old);}
   String[] lines=bytes.toString().split("\\R",-1);String x=lines[0],y=lines[1];
   check(x.length()==y.length());check(x.replace("-","").equals(a));check(y.replace("-","").equals(b));
   int actual=0,ia=0,ib=0;
   for(int i=0;i<x.length();i++) {
    char ca=x.charAt(i),cb=y.charAt(i);
    if(ca=='-') { actual+=n.penalty(ia==0 || ia==a.length());ib++; }
    else if(cb=='-') { actual+=n.penalty(ib==0 || ib==b.length());ia++; }
    else {ia++;ib++;actual+=n.score(ia,ib);}
   }
   check(score==actual);count++;
  }
  File f=File.createTempFile("empty-dna", ".txt");check(NeedlemanWunsch.load_dna(f.toString()).isEmpty());f.delete();
  NeedlemanWunsch.printmatrix("","",new int[0][0]);
  System.out.println(count+" alignment/traceback checks passed; empty file/matrix passed.");
 }
}
