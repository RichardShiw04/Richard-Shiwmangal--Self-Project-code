#include <string>
#include "Cards.h"
#include "Deck.h"
#include <cstdlib>
#include <iostream>
#include <random>
using namespace std;

/*
    TEST FUNCTIONS IN THE DECK CLASS
    Add code to test all functions in the Deck Class
*/

int main()
{
    srand(time(NULL));  // sets the seed for the random # generator algorithm 
    cout << "TEST 1: Deck(int max_size) " << endl;
    cout << "\t Declare a new Deck object my_deck of maximum size 10" << endl;
    Deck my_deck(10);
	cout << endl;

    cout << "TEST 2: int init_deck(string face[], int min_val, int max_val)" << endl;
    cout << "\t Initialize the Deck with cards of suites 'hearts' and 'spades' "
         << "\n\t with values between 1 and 5 " << endl;
    string suits[] = {"hearts", "spade"};
    my_deck.init_deck(suits, 2, 1, 5);
    cout << "\t Display the deck size" << endl; // should be 10
    cout << my_deck.get_size() << endl;
    cout << "\t Display the deck cards " << endl;
    my_deck.display();
	cout << endl;
    
    cout << "TEST 3: void shuffle(int nr_shuffles)" << endl; // shuffle all cards in the deck
    cout << "\t Shuffle the deck for 50 times" << endl;
    my_deck.shuffle(100);
    cout << "\t Display the deck" << endl;
    my_deck.display();
    cout << endl;

    cout << "TEST 4: Card peek_top() const" << endl;
    cout << "\t Display the card on the top of my_deck" << endl;
    cout << my_deck.peek_top().display() << endl;
    cout << endl;

    cout << "TEST 5: Card remove_top()" << endl;
    cout << "\t Declare a new empy deck called hand of max size 10" << endl;
    Deck hand(10);
    cout << "\t Remove the top card from my_deck and store it in a"
         <<"\n\t card object called one_card" << endl;
    Card one_card = my_deck.remove_top();
    my_deck.display();
    cout << "Card removed = " <<  one_card.display() << endl;
    cout << endl;

    cout << "TEST 5: bool is_empty() const" << endl;
    cout << "\t10. Display if hand deck is empty or not" << endl;
    cout << hand.is_empty() << endl; // this should be 1 (true)
    cout << endl;

    cout << "TEST 6: int add_top(const Card &)" << endl;
    cout << "\t Add one_card on the top of deck hand" << endl;
    hand.add_top(one_card);
    cout << "\t Display one_card" << endl;
    cout << one_card.display() << endl;
    cout << "\t Display hand" << endl;
    hand.display(); // show one card in hand
    cout << "\t Display the size of my_deck" << endl;
    cout << my_deck.get_size() << endl; // show 9 cards in my_dec
    cout << endl;

    cout << "TEST 7: int add_bottom(const Card &);" << endl;
    cout << "\t Add one_card to the bottom of my_deck" << endl;
    cout << "\t Display my_deck" << endl;
    cout << endl;

    cout << "TEST 8: Card remove(int pos)" << endl;
    cout << "\t Remove the card at position 5 from my_deck" 
         <<  "\n\t and store it in one_card" << endl;
    cout << "\t Display one_card" << endl;
    cout << "\t Add one_card to the bottom of my_deck" << endl;
    cout << "\t Display my_deck" << endl;
    cout << "\t Add one_card on the top of hand deck" << endl;
    cout << "\t Display hand" << endl;
    cout << endl;
  
    cout << "TEST 9: int get_size() const" << endl;
    cout << "\t Display hand size" << endl;
    cout << "\t Display my_deck size" << endl;
    cout << endl;

    cout << "TEST 10: bool is_full() const" << endl;
    cout << "\t Check if my_deck is full" << endl;
    cout << endl;
	
    // use a for loop
    cout << "\t Remove 5 cards from my_deck top and add them to hand top" << endl;
    cout << "\t Display my_deck: " << endl;
    cout << "\t Display hand: " << endl;
    cout << endl;

    cout << "TEST 11: void sort()" << endl;
    cout << "\t Sort hand" << endl;
    //my_deck.sort();
    cout << "\t Display hand: " << endl;
    //my_deck.display();
    cout << endl;
	
    cout << "TEST 12: int best_card(string suite) const" << endl;
    cout << "\t Display the index of the best card of suite 'hearts'" 
         << "\n\t in hand" << endl;
    cout << my_deck.best_card("hearts") << endl;
	cout << endl;

    cout << "TEST void display(int pos) const " << endl;
    cout << "\t30. Display the card at index of the best card of suite 'hearts'" 
        << "\n\t in hand" << endl;
    cout << endl;

    cout << "TEST 13: int worst_card(string suite) const" << endl;
    cout << "\t Display the index of the worst card of suite 'spades'" 
         << "\n\t in hand" << endl;
    cout << "\t Display the card at index of the worst card of suite 'spades'" 
         << "\n\t in hand" << endl;
	cout << endl;

    cout << "TEST 14: int match_face(int face) const" << endl;
    // look for a card with a particular face value that you know it is in the hand object
    cout << "\t Display the index of a face card that is in hand" << endl;
    // look for a card with a particular face value that you know it is not in the hand object
    cout << "\t Display the index of a face card that is not hand" << endl;
	cout << endl;

    cout << "TEST 15: copy constructor and assignment operator" << endl;
    Deck copy_hand(hand);
    cout << "\t Display hand " << endl;
    cout << "\t Display copy_hand " << endl;

    copy_hand = my_deck;
    cout << "\t Display deck " << endl;
    cout << "\t Display copy_hand " << endl;

}