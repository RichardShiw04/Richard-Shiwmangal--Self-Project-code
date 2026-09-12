import java.awt.*;
record coord(int y, int x) {}

/** Random depth-first maze generation, depth-first solving, and parent-path tracing. */
public class maze extends mazebase {
    

    

@Override
public void customize() {
        mwidth = 41;  // Width of the maze (must be odd)
        mheight = 41; // Height of the maze (must be odd)
        bw = bh = 18; // Block size (width and height of each cell)
        wallcolor = Color.red; // Color for maze walls
        pathcolor = Color.black;
        dotcolor = Color.white;
          // Initialize the path tracker
    }

    // 
@Override
public void digout(int y, int x)  
 {
     M[y][x]=1;
     drawblock(y,x);
     drawdot(1, 1);
     nextframe(0);
     int[] DY = {-1,0,1,0};
     int[] DX = {0,1,0,-1};
     // scramble
     for(int i=0;i<3;i++)
	 {
	     int r = i+(int)(Math.random()*(4-i));
	     int tmp = DY[i]; DY[i]=DY[r]; DY[r]=tmp;
	     tmp=DX[i]; DX[i]=DX[r]; DX[r]=tmp;
	 }//for i
     for(int d=0;d<4;d++) // for each direction
	 {
	     int nx = x+DX[d]*2, ny = y+DY[d]*2;
	     if (nx>=0 && nx<mwidth && ny>=0 && ny<mheight && M[ny][nx]==0)
		 {
		     M[y+DY[d]][x+DX[d]] = 1;
		     drawblock(y+DY[d], x+DX[d]);
		     digout(ny,nx);
		 }
	 }
 }//digout


@Override
public void trace() {
        int x = mwidth - 1;
        int y = mheight - 2;


        if (PATH == null || PATH[y][x] == null) return;

        while (!(x == 1 && y == 1)) {
            coord prev = PATH[y][x];  // Retrieve the previous step

            if (prev == null) {  // If no valid path is found
                drawMessage("Trace back failed at: (" + y + ", " + x + ")");
                drawblock(y, x);
                nextframe(0);
                return;
            }

            drawdot(y, x);
            nextframe(0);  // Visualize tracing back
            drawMessage("Now at coordinates: (" + y + ", " + x + ")");
            nextframe(0);
            delay(00);

            x = prev.x();
            y = prev.y();
        }
        y =1 ;
    x =1 ;
    drawMessage("Now at coordinates: (" + y + ", " + x + ")");
    nextframe(0);
    }

    public static void main(String[] av) {
        new maze();  // Start the maze program
    }





// public class maze extends mazebase
// {
// record coord(int y, int x) {}

    
//     // default constructor suffices and is equivalent to
public maze() { super(); }
protected coord[][] PATH ;


    
@Override
public void solve() {
    PATH = new coord[mheight][mwidth];  // Initialize PATH tracker

    int tx = mwidth - 1;
    int ty = mheight - 2;
    M[ty][tx] = 1; 
    drawblock(ty, tx);
    
    nextframe(30);

    int y = 1;
    int x = 1;
    drawdot(y, x);
    nextframe(30);

    int[] DX = {0, 1, 0, -1};
    int[] DY = {-1, 0, 1, 0};

    M[y][x]++;  // Mark the start as visited
    PATH[y][x] = new coord(y, x);

    while (!(x == tx && y == ty)) {
        int best = Integer.MAX_VALUE;
        int dir = -1;

        // Check all four directions systematically
        for (int i = 0; i < 4; i++) {
            int nx = x + DX[i];
            int ny = y + DY[i];

            if (nx >= 0 && nx < mwidth && ny >= 0 && ny < mheight && M[ny][nx] == 1) {
                if (M[ny][nx] < best) {
                    best = M[ny][nx];
                    dir = i;
                }
            }
        }

        if (dir != -1) {
            // Move to the next valid position
            int prevX = x;
            int prevY = y;
            drawblock(y, x);
            
            
            x += DX[dir];
            y += DY[dir];
            
            M[y][x]++;  // Mark as visited
            PATH[y][x] = new coord(prevY, prevX);  // Store the previous position
            
            drawdot(y, x);
            nextframe(30);

        } else {
            // Backtrack if no valid move is found
            drawblock(y, x);
            // At the root, every reachable passage has been explored.
            if (x == 1 && y == 1) {
                drawMessage("No path to exit");
                nextframe(0);
                return;
            }
            coord prev = PATH[y][x];  // Correct backtracking
            y = prev.y();
            x = prev.x();
            drawdot(y, x);  // Draw the red dot in the new position
            nextframe(30);
            
        }
    }

    // mazebase.setup() calls trace() once after solve().
    drawdot(1, 1);
     
}

}

    // //  while(!(nx>=0 && nx < mwidth && ny >= 0 && ny <mheight && M[ny][nx]==0)){
    // //     dir = (int)(Math.random()*4);
    // //     nx = x + DX[dir]*2;
    // //     ny = y + DY[dir]*2;
    // //  }
    //  M[y+DY[dir]][x+ DX[dir]] = 1;
    //  drawblock(y+DY[dir],x+DX[dir]);
    //  digout(ny, nx);
    //  }
    
    //if (x+2<mwidth && M[y][x+2]==0) // always check for maze boundaries
	 