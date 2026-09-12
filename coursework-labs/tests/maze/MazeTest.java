class MazeTest {
 public static void main(String[] args) {
  for(int run=0;run<30;run++) {
   maze m=new maze();m.digout(1,1);m.solve();
   int y=m.mheight-2,x=m.mwidth-1,steps=0;
   while(x!=1||y!=1) {
    coord p=m.PATH[y][x];
    if(p==null || Math.abs(p.y()-y)+Math.abs(p.x()-x)!=1 || ++steps>m.mwidth*m.mheight) throw new AssertionError();
    y=p.y();x=p.x();
   }
  }
  maze blocked=new maze();blocked.M[1][1]=1;blocked.solve();blocked.trace();
  if(blocked.PATH[blocked.mheight-2][blocked.mwidth-1]!=null)throw new AssertionError();
  System.out.println("30 generated maze paths and unreachable exit passed (drawing stub).");
 }
}
