#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p build/tests/java build/tests/maze build/tests/cpp
javac -d build/tests/java lab8/*.java lab7/*.java astar/*.java tests/AlignmentTest.java tests/TrieTest.java tests/AstarTest.java
java -cp build/tests/java lab8.AlignmentTest
java -cp build/tests/java lab7.TrieTest
java -cp build/tests/java astar.AstarTest
# Compile real graphics separately; algorithm tests substitute only the drawing base.
javac -d build/maze "maze lab 2/"*.java
javac -d build/tests/maze tests/maze/mazebase.java "maze lab 2/maze.java" tests/maze/MazeTest.java
java -cp build/tests/maze MazeTest
c++ -std=c++11 -Wall -Wextra -IDeckCards tests/DeckRegression.cpp DeckCards/Cards.cpp DeckCards/Deck.cpp -o build/tests/cpp/deck-regression
build/tests/cpp/deck-regression
c++ -std=c++11 -Wall -Wextra -IDeckCards DeckCards/card_game.cpp DeckCards/Cards.cpp DeckCards/Deck.cpp -o build/tests/cpp/card-game
printf 'bad\n999\n-1\n' | build/tests/cpp/card-game > build/tests/cpp/input-output.txt
printf 'All regression checks passed. GUI rendering is not covered.\n'
