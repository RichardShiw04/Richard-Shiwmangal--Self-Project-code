package lab8;

import java.io.File;
import java.io.IOException;
import java.util.Optional;
import java.util.Scanner;

/** Dynamic-programming alignment with configurable match and terminal-gap scores. */
abstract class NeedlemanWunsch {
    protected String seqA;
    protected String seqB;
    protected int[][] scoreMatrix;
    protected int[][] tracebackMatrix;

    public NeedlemanWunsch(String A, String B) {
        this.seqA = "." + A;
        this.seqB = "." + B;
        this.scoreMatrix = new int[this.seqA.length()][this.seqB.length()];
        this.tracebackMatrix = new int[this.seqA.length()][this.seqB.length()];
    }

    abstract int score(int i, int k);
    abstract int penalty(boolean edge);

    public int computeAlignment() {
        int rows = seqA.length();
        int cols = seqB.length();
        
        for (int i = 1; i < rows; i++) scoreMatrix[i][0] = i * penalty(true);
        for (int k = 1; k < cols; k++) scoreMatrix[0][k] = k * penalty(true);

        for (int i = 1; i < rows; i++) {
            for (int k = 1; k < cols; k++) {
                int match = scoreMatrix[i - 1][k - 1] + score(i, k);
                int delete = scoreMatrix[i - 1][k] + penalty(k == cols - 1);
                int insert = scoreMatrix[i][k - 1] + penalty(i == rows - 1);
                scoreMatrix[i][k] = Math.max(match, Math.max(delete, insert));

                if (scoreMatrix[i][k] == match) tracebackMatrix[i][k] = 0;
                else if (scoreMatrix[i][k] == delete) tracebackMatrix[i][k] = 1;
                else tracebackMatrix[i][k] = 2;
            }
        }
        return scoreMatrix[rows - 1][cols - 1];
    }

    /** Prints the alignment after computeAlignment(); boundary gaps consume only the nonempty sequence. */
    public void traceback() {
        StringBuilder alignedA = new StringBuilder();
        StringBuilder alignedB = new StringBuilder();
        int i = seqA.length() - 1;
        int k = seqB.length() - 1;

        while (i > 0 || k > 0) {
            if (i > 0 && k > 0 && tracebackMatrix[i][k] == 0) {
                alignedA.append(seqA.charAt(i));
                alignedB.append(seqB.charAt(k));
                i--; k--;
            } else if (i > 0 && (k == 0 || tracebackMatrix[i][k] == 1)) {
                alignedA.append(seqA.charAt(i));
                alignedB.append("-");
                i--;
            } else {
                alignedA.append("-");
                alignedB.append(seqB.charAt(k));
                k--;
            }
        }
        System.out.println(alignedA.reverse());
        System.out.println(alignedB.reverse());
    }

    public static String random_dna(int n) {
        if (n < 1) return "";
        char[] C = new char[n];
        char[] DNA = {'A', 'C', 'G', 'T'};
        for (int i = 0; i < n; i++)
            C[i] = DNA[(int) (Math.random() * 4)];
        return new String(C);
    }
    
    public static Optional<String> load_dna(String filename) {
        Optional<String> answer = Optional.empty();
        try (Scanner br = new Scanner(new File(filename))) {
            if (br.hasNextLine()) answer = Optional.of(br.nextLine());
        } catch (IOException ie) {
            ie.printStackTrace();
        }
        return answer;
    }
    
    public static void printmatrix(String A, String B, int[][] M) {
        if (A == null || B == null || M == null || M.length == 0 || M[0] == null) return;
        if (A.length() != M.length || B.length() != M[0].length) return;
        int rows = A.length();
        int cols = B.length();
        System.out.print("    ");
        for (int i = 0; i < cols; i++) System.out.printf(" %2s ", B.charAt(i));
        System.out.println();
        for (int i = 0; i < rows; i++) {
            System.out.print("   " + A.charAt(i));
            for (int k = 0; k < cols; k++) {
                System.out.printf(" %2d ", M[i][k]);
            }
            System.out.println();
        }
    }
}

class Scheme1 extends NeedlemanWunsch {
    public Scheme1(String A, String B) { super(A, B); }

    int score(int i, int k) {
        return (seqA.charAt(i) == seqB.charAt(k)) ? 1 : 0;
    }

    int penalty(boolean edge) { return 0; }
}

class Scheme2 extends NeedlemanWunsch {
    public Scheme2(String A, String B) { super(A, B); }

    int score(int i, int k) {
        return (seqA.charAt(i) == seqB.charAt(k)) ? 1 : -1;
    }

    int penalty(boolean edge) { return -1; }
}

class Scheme3 extends NeedlemanWunsch {
    public Scheme3(String A, String B) { super(A, B); }

    int score(int i, int k) {
        return ((seqA.charAt(i) | 32) == (seqB.charAt(k) | 32)) ? 3 : -1; // Case-insensitive
    }

    int penalty(boolean edge) {
        return edge ? 0 : -2;
    }
}
