import java.awt.*;
import java.awt.event.*;
import java.awt.Graphics;
import javax.swing.*;

/**
* <h2>Base class of maze generation/solution assignments for CSC17.</h2>
*
* <b>Protected fields, except the 2D array {@link M}, can be modified by 
* overriding the {@link customize} function in a subclass.
*<br>
* This class does not contain a main, which should be supplied in a subclass.
* </b>
* @author Chuck Liang
* 
*/
public class mazebase extends JFrame implements KeyListener
{
/** <b>The number of rows of the maze 2D array.  This number should be odd in order for the maze to have a border.</b> */
protected int mheight = 41;
/** <b>The number of columns of the maze 2D array.  This number should be odd in order for the maze to have a border.</b> */    
protected int mwidth = 41;
/** 
<b>The array for the maze is declared and initialized in the superclass.</b>  This
variable should be not set in {@link customize}
 */
protected int[][] M;	// the array for the maze
    /*
public static final int SOUTH = 0;
public static final int EAST = 1;
public static final int NORTH = 2;
public static final int WEST = 3;
    */
    /** If set to true, the {@link drawblock} function will display the values of the maze matrix that are non-zero */
protected boolean showvalue = false; // affects drawblock
    /** If set to true, will automatically delay by value {@link dtime} at end of
calls to {@link drawdot} */    
protected boolean autodelay = false;  //delays automatically between drawdot
    /** If set to true, will try to load animated gif from filename {@link gifname}, which will be displayed by {@link drawdot} */
protected boolean usegif = false;     // affects drawdot
    
// graphical properties:
/** the height in pixels of graphical square representing a maze coordinate */    
protected int bh = 20; 	// height of a graphical block
/** the width in pixels of graphical square representing a maze coordinate */        
protected int bw = 20;	// width of a graphical block
private int ah, aw;	// height and width of graphical maze (don't change)
    /** "y-offset", which is required on some systems that automatically displays a title bar for every window. */
protected int yoff = 40;    // init y-cord of maze
private Image screen1;    // static background buffer
private Image screen2;    // animation frame buffer
private Graphics g;  // draw to screen1
/**  returns java.awt.Graphics object that allows drawing to animation
buffer: nothing will be displayed, however, until {@link nextframe} is
called.
*/
protected Graphics Graphics() {return g;}    
private Graphics g2;  // draw to screen2    
private Graphics dg; // draw to actual display
    /** The approximate number of milliseconds delayed in calls to {@link drawdot} if {@link autodelay} is set to true */
protected int dtime = 30;   // ms delay time (for autodelay)
    /** The color of maze "walls", or M[row][column] values that are zero */
protected Color wallcolor = Color.green;
    /** The color of maze "paths", or M[row][column] values that are non-zero */   
protected Color pathcolor = Color.black;
    /** The color of the graphical circle drawn by default by {@link drawdot} */
protected Color dotcolor = Color.red;
    /** The color of text messages that are displayed on the maze by {@link drawMessage}*/
protected Color pencolor = Color.yellow;    
private Image animatedgif;
    /** The path-name of the gif/jpeg file that replaces a circle if {@link usegif} is set to true */
protected String gifname = "miner.gif";
    /** Starting x coordinate of maze */
protected int startdigx = 1;  // initial coordinates, first call to digout
    /** Starting y coordinate of maze: the mazebase superclass will call 
{@code digout(startdigy,startdigx)} */    
protected int startdigy = 1;    


    /** constructor calls customize first, then creates the matrix M
     */
 public mazebase()
 { 
     //   bh = bw = bh0;  mheight = mh0;  mwidth = mw0;
   customize(); // optional startupcode  - change all vars here          
   ah = bh*mheight;
   aw = bw*mwidth;
   M = new int[mheight][mwidth];  // initialize maze (all  0's - walls).
   this.setBounds(0,0,aw+10,10+ah+yoff);	
   this.setVisible(true);
   this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
   this.addKeyListener(this);
   try{Thread.sleep(500);} catch(Exception e) {} // Synch with system
   screen1 = createImage(aw+10,10+ah+yoff);
   screen2 = createImage(aw+10,10+ah+yoff);   
   g = screen1.getGraphics();
   g2 = screen2.getGraphics();   
   dg = getGraphics();    //g.setColor(Color.red);
   setup();
 }

    // utility to load animated gif, called from setup after customize()
private void loadgif(String filename)
 {
     try {
      animatedgif = Toolkit.getDefaultToolkit().getImage(filename);
      prepareImage(animatedgif,this);
      Thread.sleep(100); // Synch with system
     } catch(Exception e) {animatedgif=null; usegif=false;} 
 }//loadgif
    
    /** this function overrides automatic repaint: do not change */
@Override    
public void paint(Graphics g) {} // override automatic repaint

private void setup()
   {
     g.setColor(wallcolor);
     g.fill3DRect(0,yoff,aw,ah,true);  // fill raised rectangle
     g.setColor(pathcolor);
     try {
        g.setFont(new Font("Serif",Font.BOLD,bh*3/4));      // might not work
     } catch(Exception gfe) {}
     /*if (usegif)*/ loadgif(gifname); // placed here so user can define gif usage
     //     showStatus("Generating maze...");
     //digout(mheight-2,mwidth-2); // start digging!  (lab 2)
     digout(startdigy,startdigx);  
     // draw to screen
     clear();
     refresh();
     /*
     // digout exit square (if digout complete, works for odd dimensions)
     if (M[mheight-2][mwidth-2]!=0)
	 {
	     //M[mheight-1][mwidth-2] =
	     M[mheight-2][mwidth-1] = 1;
	     drawblock(mheight-2,mwidth-1);
	 }
     */
     solve();  // this is the function to @Override for lab 3, part 1
     trace();  // for part 2
     play();   // for part 3
   }   


    protected void clear()
    {
        g2.drawImage(screen1,0,0,this);        //draw to screen2
    }
    protected void refresh()
    {
     dg.drawImage(screen2,0,0,this);             
    }
    /** redraws animation buffer without delay */
    public void nextframe()
    { clear(); refresh(); }
    /** <b>redraws animation buffer after specified delay </b>
     *  @param msdelay The approximate number of milliseconds to delay
    */
    public void nextframe(int msdelay) // with delay time in ms
    { delay(msdelay); clear(); refresh(); }

    /** delays current thread by approximately ms milliseconds.  This
        function is non-blocking.
     */    
public void delay(int ms)
    {   
	try {Thread.sleep(ms);} catch(Exception e) {}
    }

    /**
       <b>draws a graphical square of color pathcolor corresponding to matrix
       coordinates row y, column x.</b>
     */
public void drawblock(int y, int x)
    {
        drawblock(g,y,x);
    }
private void drawblock(Graphics g,int y, int x)
    {
	g.setColor(pathcolor);
	g.fillRect(x*bw,yoff+(y*bh),bw,bh);
	g.setColor(pencolor);
	// following line displays value of M[y][x] in the graphical maze:
	if (showvalue)
	  g.drawString(""+M[y][x],(x*bw)+(bw/2-4),yoff+(y*bh)+(bh/2+6));
    }    

    /** draws a solid circle of color {@link dotcolor} corresponding to matrix
        coordinates y,x, unless {@link usegif} is set true, in which case it
will draw a gif as indicated by {@link gifname} */
public void drawdot(int y, int x)
    {
        drawdot(g,y,x);
    }
protected void drawdot2(int y, int x) {drawdot(g2,y,x);}    
private void drawdot(Graphics g, int y, int x)
    {
	if (usegif && animatedgif!=null)
	    {
		g.drawImage(animatedgif,x*bw,yoff+(y*bh),bw,bh,null);
	    }
	else
	    {
		g.setColor(dotcolor);
		g.fillOval(x*bw,yoff+(y*bh),bw,bh);
	    }
        if (autodelay) try{Thread.sleep(dtime);} catch(Exception e) {} 
    }
    /** draw default gif */
public void drawgif(int y, int x) { drawgif(g2,animatedgif,y,x); }  //alias
    /** draw a specific Image (such as animated gif) */
protected void drawgif2(int y,Image gif,int x) { drawgif(g2,gif,y,x);}  

private void drawgif(Graphics g, Image gif, int y, int x)
    {
        g.drawImage(gif,x*bw,yoff+(y*bh),bw,bh,null);        
    }

    /** displays string message m at top row of maze */
public void drawMessage(String m)
    {
	g.setColor(wallcolor);
	g.fillRect(0,yoff,bw*mwidth,bh);
	g.setColor(pencolor); // erase line
        g.drawString(m,10,yoff+bh-4);	
    }

////// the following functions are to be overriden in subclass:

    /** User-defined initialization code.  This function can be overridden in 
a subclass. The default, base class version does nothing.  This function is
called by the constructor of base class */
public void customize()  // user-defined initialization code
{} // this is called before digout

/* function to generate random maze */

    /** <b>This function must be override in a subclass to generate a random
maze.  The default, base-class version does nothing.</b>
     */
public void digout(int y, int x)    // override for lab 2 (maze generation)
 {
     // generates maze - code in subclass
 } // digout


    /** Override this function in a subclass to solve the maze.
       Start at coordinates x=1, y=1, and stop at coordinates
       x=mwidth-1, y=mheight-2.
    */
  public void solve()    // override for lab 3 part 1
  {
      //int x=1, y=1;
      //      drawdot(y,x);
      // drawblock(y,x) will erase the dot
      // modify this function to move the dot to the end of the maze.  That
      // is, when the dot reaches y==mheight-2, x==mwidth-2
  } // solve

    /**  Override this function to trace the optimal solution path of the
maze. 
     */
  public void trace()     // override for lab 3 part 2, 
  {  // draw a dot (without erasing it) along the OPTIMAL path
  }

    ///////////////////////////////////////////////////////////////
    /// For part three (save a copy of part 2 version first!), you
    // need to implement the KeyListener interface.

    /** Optional function to override to set up an interactive game */
    public void play() // override for lab 3, final part
    {
	// code to setup game
    }
    // for this part you may also define some other instance vars outside of
    // the play function.

   // skeleton implementation of KeyListener interface
   public void keyReleased(KeyEvent e) {}
   public void keyTyped(KeyEvent e) {}
    /** Override this function to respond to keyboard events.  The
default function prints the numerical value corresponding to the key 
pressed */
   public void keyPressed(KeyEvent e) // override for key event handling
    {
	int key = e.getKeyCode();       // code for key pressed      
	System.out.println("YOU JUST PRESSED KEY "+key);
    }
    
    /*
    // main:   
  public static void main(String[] args) throws Exception
    {
       int n = args.length;
       if (n==1 || n==4) subclass = args[0];
       int blocksize = bh, mh = 45, mw = 45; // width/height need to be odd
       if (n==4 || n==3)
	   {
	       mh=Integer.parseInt(args[n-3]);
	       mw=Integer.parseInt(args[n-2]);
	       blocksize=Integer.parseInt(args[n-1]);
	   }
       //       mazebase W = new mymazecode(blocksize,mh,mw);
    }//main
   */

    // main should be in subclass
} // mazebase