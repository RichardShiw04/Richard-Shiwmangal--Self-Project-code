package astar;
import java.util.*;
public class AstarTest {
 public static void main(String[] args) {
  Random rng=new Random(42);
  for(int run=0;run<100;run++) {
   myastar a=new myastar(8,8);a.setcosts(1,-1,5,run % 2 == 0 ? 0 : 2);
   for(int[] row:a.Map)for(int x=0;x<row.length;x++)row[x]=rng.nextInt(4);
   a.Map[0][0]=0;
   int[][] d=new int[8][8];for(int[] row:d)Arrays.fill(row,100000);d[0][0]=0;
   for(int pass=0;pass<64;pass++)for(int y=0;y<8;y++)for(int x=0;x<8;x++)for(int dir=0;dir<6;dir++) {
    int ny=y+astar_base.DY[dir],nx=x+astar_base.DX[y%2][dir];
    if(ny>=0&&ny<8&&nx>=0&&nx<8&&a.costof[a.Map[ny][nx]]>=0)
     d[ny][nx]=Math.min(d[ny][nx],d[y][x]+a.costof[a.Map[ny][nx]]);
   }
   Optional<coord> p=a.search(0,0,7,7);
   if(p.isPresent() ? p.get().known_cost()!=d[7][7] : d[7][7]<100000)throw new AssertionError();
   if(a.search(-1,0,7,7).isPresent())throw new AssertionError();
  }
  System.out.println("100 A* weighted maps matched independent shortest-path costs; invalid coordinates passed.");
 }
}
