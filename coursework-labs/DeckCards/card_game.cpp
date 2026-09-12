#include <string>
#include "Cards.h"
#include "Deck.h"
#include <cstdlib>
#include <ctime>
#include <limits>
#include <random>
#include <iostream>
using namespace std;

int main()
{
    srand(time(NULL));
    // 1. Instantiate an empty Deck object called deck with a max_size of 52
    Deck deck(52);
    // 2. Initialize the deck with 52 cards from 
    // valid suites: {"hearts", "diamonds", "spade", "clubs"}
    // range of values 1 to 13 
    string suites[] = {"hearts", "diamonds", "spade", "clubs"};
    
    deck.init_deck(suites,4,1,13);
    // 3. Shuffle the deck
    deck.shuffle(500);
    // deck.display(); // tests ok

   // Simulate a simple UNO like game

    // 4. Instantiate two empty Deck objects called my_hand and 
    // computer_hand with a max_size of 52
    Deck my_hand(52),
         computer_hand(52);

    // 5. Instantiate a new empty Deck of max_size 52 called pile
    Deck pile(52);
    
    // 6. Remove the top Card from the deck and add it to my_hand
    //    Remove the next top Card from deck and add it to the computer_hand
    //    Repeat this seven times, until both my_hand and computer_hand
    //    each contain 7 cards
    for (int i = 0; i < 7; i++)
    {
        my_hand.add_top(deck.remove_top());
        computer_hand.add_top(deck.remove_top());
    } 
   // my_hand.display(); // test ok
   // computer_hand.display(); // test ok
    my_hand.sort();
    computer_hand.sort();

    // 7. Simulate a simple game betweem computer_hand and my_hand:
       // remove card from top of the deck card and add it to the top
       // of the pile
        pile.add_top(deck.remove_top());
       // choose randomly who starts first: 1 for you, 0 for computer
       int turn;
       turn = rand()%2; 
       // while deck is not empty or none of hands are empty:
       bool game_over = false; // whenever one player has an empty hand - finish the game
       
        while (!game_over)
        {
          // display card on top of the pile
          cout << "Card on top of the pile " << endl;
          cout << pile.peek_top().display() << endl;

          // display the player's turn hand
          
          switch(turn){
              case 1: cout << "Your cards: " << endl;  // your turn
                      my_hand.display(); 
                      break;
              case 0: cout << "Computer cards:" << endl; // computer turn 
                      computer_hand.display();
          }
          // the player who's turn is chooses a card that matches either 
          // the suite or the value of the card on top of the pile
          // or draws a new card from top of the deck

          // If it is your turn, you can chose the card you want to play or draw
          // a new Card from the deck. 
          int ind_card = -1;
          Card chosen_card;
          switch(turn)
          {
              case 1: 
                    cout << "Enter the index of the card you want to play between 0 and " << 
                           my_hand.get_size()-1 << " or -1 to pick from the deck" << endl;
                    if (!(cin >> ind_card)) {
                        if (cin.eof()) return 0;
                        cin.clear();
                        cin.ignore(numeric_limits<streamsize>::max(), '\n');
                        cout << "Enter a card index or -1." << endl;
                        continue;
                    }
                    if (ind_card < -1 || ind_card >= my_hand.get_size()) {
                        cout << "Invalid card index." << endl;
                        continue;
                    }
                    if (ind_card == -1){
                        chosen_card = deck.remove_top();
                        my_hand.add_top(chosen_card);
                        cout << "You drew a card from the deck" << endl;
                    }
                    else{
                        chosen_card = my_hand.remove(ind_card);
                        if (!chosen_card.same_suite(pile.peek_top()) &&
                            chosen_card.get_face() != pile.peek_top().get_face()) {
                            my_hand.add_top(chosen_card);
                            cout << "Match the suit or face of the pile card." << endl;
                            continue;
                        }
                        pile.add_top(chosen_card);
                        if (my_hand.get_size() == 1)
                            cout << " Your UNO" << endl;
                    }
                    break;
              case 0: // computer choses a card 
                    ind_card = computer_hand.best_card(pile.peek_top().get_suite());
                    if (ind_card == -1)
                        ind_card = computer_hand.match_face(pile.peek_top().get_face());
                    if (ind_card == -1){
                        chosen_card = deck.remove_top();
                        computer_hand.add_top(chosen_card);
                        cout << "Computer drew a card from the deck" << endl;
                    }
                    else{
                        chosen_card = computer_hand.remove(ind_card);
                        pile.add_top(chosen_card);
                        if (computer_hand.get_size() == 1)
                            cout << "Computer UNO" << endl;
                    }
          }
        
          // If it is computer's turn, use the functions in Deck class to 
          // either chose a card or draw a Card from the deck.  
      
          // if player has one card left in their hand display UNO

          // if any player chooses to place a card, add that card on top of the pile.

          // change turns
          turn = (turn+1) % 2;
          // check if game over
          game_over = (deck.is_empty() || my_hand.is_empty() || computer_hand.is_empty());
        }
        
       // Display result of the game
       if (deck.is_empty())
         cout << "Deck was empty" << endl;
       else if (my_hand.is_empty())
         cout << "You won!" << endl;
       else
         cout << "Computer won" << endl;

}