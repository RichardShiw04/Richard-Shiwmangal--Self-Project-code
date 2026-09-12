import java.awt.Color;
class mazebase {
 protected int mwidth,mheight,bw,bh; protected Color wallcolor,pathcolor,dotcolor; protected int[][] M;
 mazebase(){customize();M=new int[mheight][mwidth];}
 public void customize(){} public void digout(int y,int x){} public void solve(){} public void trace(){}
 public void drawblock(int y,int x){} public void drawdot(int y,int x){} public void nextframe(int n){}
 public void delay(int n){} public void drawMessage(String s){}
}
