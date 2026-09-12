package astar;

import java.util.Optional;

// don't change name of your class (referred to in pathfinder.java)
/** A* search on the assignment hex grid using a hashed minimum heap. */
public class myastar extends astar_base
{
    public myastar(int r, int c)
    { super(r,c); }

    @Override
    public void customize() {
       ////// Things you can do here...
        setcosts(1,1,1,1); // cost of land, desert, fire, water
        //pathfinder.gap = 15; // change size of graphical hexgagons
        //pathfinder.yoff = 20; // graphical top margin adjustment
        //pathfinder.delaytime = 300; //change animation speed
        //setRandFactor(0.13); // increase amount of water/fire
    }
    public static void main(String[] av) {
        pathfinder.main(av);
    }//main

    ///// The following function just searches randomly.
    ///// You must replace it with an implementation of Algorithm A*:
    // @Override
    // public Optional<coord> search(int sy, int sx, int ty, int tx)  {
    //     coord current = new coord(sy,sx);
    //     while (current.y!=ty || current.x!=tx)  // solve by random search!
    //         {                                   // worst case O(infinity)
    //             //pick random direction
    //             int dir = (int)(Math.random()*6);
    //             int cy = current.y, cx = current.x;
    //             int ny = cy + DY[dir]; 
    //             int nx = cx + DX[cy%2][dir];
    //             if (nx>=0 && nx<COLS && ny>=0 && ny<ROWS)
    //                 {
    //                     coord next = new coord(ny,nx);
    //                     next.parent = Optional.of(current);
    //                     current = next; 
    //                 }
    //             // else, loop back and pick another direction
    //         }// main while
    //     return Optional.of(current);
    // }//search


    



    @Override
public Optional<coord> search(int sy, int sx, int ty, int tx) {
    if (sy < 0 || sy >= ROWS || sx < 0 || sx >= COLS ||
        ty < 0 || ty >= ROWS || tx < 0 || tx >= COLS) return Optional.empty();
    HashedHeap.set_initial_capacity(ROWS * COLS);
    var Frontier = new HashedHeap<Integer, coord>((a, b) -> b.compareTo(a));
    boolean[][] Interior = new boolean[ROWS][COLS];

    // Zero-cost terrain requires a zero heuristic to remain admissible.
    boolean zeroCost = java.util.Arrays.stream(costof).anyMatch(c -> c == 0);

    // Initialize the start node
    coord start = new coord(sy, sx);
    start.set_known_cost(0);
    start.add_estimated_cost(zeroCost ? 0 : hexdist(sy, sx, ty, tx));
    Frontier.push(hash_key(sy, sx), start);

    while (Frontier.size() > 0) {
        Optional<KVPair<Integer, coord>> popped = Frontier.pop();
        if (popped.isEmpty()) {
            return Optional.empty(); // Shouldn't occur, but safe check
        }
        coord current = popped.get().val(); // Extract coord

        // If the target is reached
        if (current.y == ty && current.x == tx) {
            return Optional.of(current);
        }

        Interior[current.y][current.x] = true; // Mark current as visited

        // Explore neighbors
        for (int dir = 0; dir < 6; dir++) {
            int ny = current.y + DY[dir];
            int nx = current.x + DX[current.y % 2][dir];

            if (nx >= 0 && nx < COLS && ny >= 0 && ny < ROWS &&
                costof[Map[ny][nx]] >= 0 && !Interior[ny][nx]) {

                coord neighbor = make_neighbor(current, ny, nx, ty, tx);
                if (zeroCost) neighbor.add_estimated_cost(0);

                // Check if already in Frontier and update if necessary
                var existing = Frontier.get(hash_key(ny, nx));
                if (existing.isPresent()) {
                    if (neighbor.compareTo(existing.get()) < 0) {
                        Frontier.and_modify(hash_key(ny, nx), oldOpt -> neighbor);
                    }
                } else {
                    Frontier.push(hash_key(ny, nx), neighbor);
                }
            }
        }
    }
    return Optional.empty(); // No path found
}
}
//myastar

/*  More Help:
  
   Algorithm A* is a modification of Dijkstra's Algorithm with the following
   differences:
  
   1. Instead of finding shortest paths form the source to all destinations,
      A* is focused on finding the best path from the source one specific
      destination.  The source is given by coordinates sy,sx and the 
      destination by ty,tx.  The algorithm terminates when the destination
      node (coord object) has been removed from the frontier and inserted 
      into the interior.

   2. The cost of each node in the search tree is a sum of two elements:
        
             known_cost_from_src  +  estimated_cost_to_dst

      Where the known_cost_from_src is the same as the cost measure used
      in Dijkstra's algorithm.  The estimated_cost_to_dst is a heuristic
      estimate of the remaining cost from the current node to the
      destination.  Futhermore, the estimate must be conservative: it
      cannot exceed the actual cost.  For example, when finding the best
      route to drive to a certain destination, the estimate can be the
      straight-line distance to the destination, which is guaranteed to 
      not overestimate the actual distance.  Under this restriction, A*
      is guaranteed to also find the optimal path to target.  

      The `hexdist` function in the astar_base superclass provides a
      conservative estimate of the remaining cost to reach the target.

   You can also think of Dijkstra's algorithm as Algorithm A* with a 
   heuristic estimate of zero. 
   
   As for data structures, I suggest you use my HashedHeap for the 
   Frontier.  You can create an instance of it as follows:

     HashedHeap.set_initial_capacity(ROWS*COLS); //never have to resize/rehash
     var Frontier = new HashedHeap<Integer,coord>((x,y)->y.compareTo(x));

   This creates a minheap.  The keys of the HashedHeap are integers
   computed from the y,x coordinates of a coord object, specifically
   y*COLS+x: the one-dimensional representation of the 2D coordinate.
   If given initial capacity ROWS*COLS, this guarantees that there
   will be no hash-collisions, and no need to ever resize the
   structure.

   For the Interior, you can use a hashmap but I would suggest just using 
   a 2D array of booleans:

     boolean[][] Interior = new boolean[ROWS][COLS];

   Initially, each Interior[y][x]==false, until you set it otherwise.

   Refer to my notes on Dijkstra's Algorithm for pseudocode.
*/
