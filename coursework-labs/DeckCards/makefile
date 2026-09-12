
# from: https://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
all: card_game test_deck
CC=g++
CFLAGS=-I. -std=c++11 -Wall -Wextra
DEPS = Deck.h Cards.h  # dependencies 
OBJ  = test_deck.o  card_game.o Cards.o Deck.o # name of object files 
# compile all cpp files to an object file
%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)   
# -c compile 
# -o = output file (.o file) 
# $@ = item on the left of : (%.o)
# $< = first item in the list after : (%.cpp)

# link all object files into an executable file called card_game
card_game: card_game.o Cards.o Deck.o
	$(CC) -o $@ $^ $(CFLAGS)
# $@ = left side of : (card_game)
# $^ = right side of : $(OBJ)

# link all object files into an executable file called test_deck
test_deck: test_deck.o Cards.o Deck.o
	$(CC) -o $@ $^ $(CFLAGS)
# $@ = left side of : (card_game)
# $^ = right side of : $(OBJ)

# delete executable files
clean: 
	$(RM) card_game
	$(RM) test_deck
	$(RM) *.o
