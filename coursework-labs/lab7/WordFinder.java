package lab7;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.*;

public class WordFinder extends JFrame implements KeyListener {
    private Graphics display;
    private StringTrie<Integer> rwords;
    private int XDIM = 800;
    private int YDIM = 600;
    private int xoff = 50;
    private int yoff = 50;
    private String resultMessage = "";
    private StringBuilder currentWord;  // To hold the current word being typed

    // Constructor
    public WordFinder() {
        // Initialize the frame and other components
        setTitle("Word Finder");
        setSize(XDIM, YDIM);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);


        // Add KeyListener to capture key presses
        addKeyListener(this);
        setFocusable(true); // Ensure the JFrame can capture key events
        
        // Initialize the trie (or some other collection for storing words)
        rwords = new StringTrie<>();

        // Initialize the currentWord StringBuilder
        currentWord = new StringBuilder();
        
        // Load words from file
        loadWordsFromFile("hfwords.txt");

        // Start the game or logic
        setVisible(true);
        newframe();
    }

    // Method to load words from a file (hfwords.txt)
    public void loadWordsFromFile(String filename) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                // Check the method to add the word to the trie
                String word = line.trim().replaceFirst("^\\d+\\s*", "").toLowerCase(Locale.ROOT);
                if (!word.isEmpty()) rwords.add(word, 1);
            }
        } catch (IOException e) {
            System.out.println("Error loading words from file: " + e.getMessage());
        }
    }

    // Method to create a new frame (clear the screen)
    public void newframe() {
        repaint();
    }

    // Method to display the current word on the screen
    public void displayWord() {
        display.setColor(Color.black);
        display.drawString("Current word: " + currentWord.toString(), xoff + 10, YDIM - 100);
    }

    // Paint method for drawing on the JFrame
    @Override
    public void paint(Graphics g) {
        super.paint(g);
        display = g;  // Assign graphics object for painting
        if (currentWord == null) return;
        g.setColor(Color.white);
        g.fillRect(0, 0, XDIM, YDIM);
        g.setColor(Color.blue);
        g.drawString("Word Finder — Enter to search, Space to clear", xoff, 50);
        displayWord();
        g.drawString(resultMessage, xoff + 10, YDIM - 120);
    }

    // Method to search for a word and display the result
    public void searchWord() {
        String word = currentWord.toString();
        resultMessage = rwords.get(word) != null ? "Found word: " + word : "Word not found: " + word;
        repaint();
    }

    // KeyListener method for handling key events
    @Override
    public void keyTyped(KeyEvent e) {
        char keyChar = e.getKeyChar();
        if (Character.isLetter(keyChar)) {
            // Add the typed character to the current word
            currentWord.append(Character.toLowerCase(keyChar));
            resultMessage = "";
        }
        // Check for Enter key to search the word
        if (e.getKeyChar() == KeyEvent.VK_ENTER) {
            searchWord();
        }
        // Check for Space key to reset the current word
        if (e.getKeyChar() == KeyEvent.VK_SPACE) {
            currentWord.setLength(0); // Reset the current word
            resultMessage = "";
        }
        newframe();  // Update the display after each key press
    }

    @Override
    public void keyPressed(KeyEvent e) {
        // Not used in this example, but can be used for more advanced key handling
    }

    @Override
    public void keyReleased(KeyEvent e) {
        // Not used in this example
    }

    // Method to reset the game or clear the screen
    public void resetGame() {
        rwords.clear();  // Clear the trie
        loadWordsFromFile("hfwords.txt");  // Reload words if needed
        currentWord.setLength(0);  // Reset the current word
        newframe();  // Refresh the frame
    }

    // Main method to run the application
    public static void main(String[] args) {
        SwingUtilities.invokeLater(WordFinder::new);
    }
}
