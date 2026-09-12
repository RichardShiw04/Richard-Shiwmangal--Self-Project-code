#include <string>
#include "Cards.h"
#include "Deck.h"
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <stdexcept>
#include <random> 
using namespace std;

/*
    Goal: initialize an empty deck of max_size
    Inputs: max_size
    Outputs: none
    Processing: 
        Allocate dynamic memory for an array of Card objects of max_size
        ptr_cards = point to the new array
        nr_cards = 0
        max_size = max_deck_size
    Preconditions: max_deck_size > 0 
    Postconditions: dynamic memory for a new deck of max_size is allocated
                    no cards in the deck
*/
Deck::Deck(int max_deck_size) // empty deck of max_size
{
  assert(max_deck_size > 0);
  max_size = max_deck_size;
  nr_cards = 0;
  ptr_cards = new Card[max_size];
}

/*
  Copy constructor: instantiate a new Deck object 
  as a deep copy of an old Deck object
  Initialize the new Deck as an empty Deck of the same max_size 
  as other Deck. 
  Then call the assignment operator
*/
Deck::Deck(const Deck &other)
{
    max_size = other.max_size;
    nr_cards = other.nr_cards;
    ptr_cards = new Card[other.max_size]; // allocate dynamic memory

    //copy all cards from other Deck to the new Deck
    for (int i = 0; i < other.nr_cards; i++)
        ptr_cards[i] = other.ptr_cards[i];
}

 // destructor - deallocates dynamic memory in the Deck object
 // deallocate memory for the dynamic array pointed by ptr_cards
Deck::~Deck()
{
    delete [] ptr_cards;
}

// assignment operator
// make a deep copy of the other Deck
Deck & Deck::operator=(const Deck &other)
{
    // check against self-assignment
    if (this == &other)
        return *this;

    // 1. deallocate memory for the Cards array in this object
    delete [] ptr_cards;
    // 2. allocate memory equal to other.max_size
    ptr_cards = new Card[other.max_size];
    // 3. copy the other.nr_cards Cars from other to this deck
    for (int i = 0; i < other.nr_cards; i++)
        ptr_cards[i] = other.ptr_cards[i];
    max_size = other.max_size;
    nr_cards = other.nr_cards;

    return *this;
}

/*
    Goal: initialize deck with Cards
    Inputs: a string of suits, num_suits = number of suits, min_val and max_val
    Outputs: number of cards added to the deck
    Processing:
        For each face in suits
            For each value in min_val to max_val
                - Create a new Card object
                - add it in ptr_cards at position nr_cards
                - increment nr_cards
    Preconditions: nr_cards = 0
    Postconditions: a number of cards with suites in the string and values 
    in the range min_val and max_val are added to the deck
*/
int Deck::init_deck(string suits[], int num_suits, int min_val, int max_val)
{
    assert(nr_cards == 0);
    // Reject ranges that would overflow the allocated card array.
    if (num_suits < 0 || min_val < 1 || max_val > 14 || min_val > max_val ||
        static_cast<long long>(num_suits) * (max_val - min_val + 1) > max_size)
        throw std::invalid_argument("Card range exceeds deck capacity or valid faces");

    for (int s = 0; s < num_suits; s++){ // for each suit string in array suits
        for (int v = min_val; v <= max_val; v++){ // for each face value in range min_val to max_val
            ptr_cards[nr_cards].set_face(v);
            ptr_cards[nr_cards].set_suite(suits[s]); 
            nr_cards++;
        }
    }
    return nr_cards;
}

/*
    Goal: shuffle the deck of cards
    Inputs: an integer with number of shuffles
    Outputs: none
    Processing:
        Your choice: this is a suggestion
        repeat num_shuffles:
            pick a random position between 0 and nr_cards -1
            swap it with a random position betweem 0 and nr_cards-1     
    Preconditions: none, 
    Postconditions: cards in the deck are shuffled
*/
void Deck::shuffle(int num_shuffles) // shuffle all cards in the deck
{
    if (nr_cards < 2) return; // Avoid modulo zero or an endless self-swap loop.
    int shuffles = 0; 
    // while loop shuffles is less than num_shuffles 
    while (shuffles < num_shuffles)
    {
        // select two random card indices between 0 and nr_cards-1 .
        int ind1 = rand() % nr_cards;
        int ind2 = rand() % nr_cards;
        // don't count as a shuffle if ind1 is equal to ind2 
        if (ind1 == ind2)
            continue; 
        // swap the cards (use a temp Card object)
        Card temp = ptr_cards[ind1];
        ptr_cards[ind1] = ptr_cards[ind2];
        ptr_cards[ind2] = temp;
        
        shuffles++;
    }
}

/*
    Goal: add new_card to top of the Deck
    Inputs: reference to a constant new_card object
    Outputs: 0 if not successful, 1 if successful
    Processing:
        Top of the deck is at index nr_cards-1
        add new_card to ptr_cards at index nr_cards
        increment nr_cards
    Preconditions: deck is not full, nr_cards is less max_size   
    Postconditions: a new card is added to the deck
*/
int Deck::add_top(const Card &new_card)
{
    if (is_full())
        return 0;

    ptr_cards[nr_cards] = new_card;
    nr_cards++;
    return 1;
}

/*
    Goal: add a new card to the deck - at the bottom of the Deck
    Inputs: a reference to a constant card object:new_card
    Outputs: 0 if not successful, 1 if successful
    Processing:
        Bottom of the deck is at index 0
        - move all cards from index 0 to nr_cards-1 one position
         towards the end of the array
        - add new_card at index 0
        - increment nr_cards
    Preconditions: deck is not full, nr_cards is less max_size
    Postconditions: a new card is added to the bottom of the deck
*/
int Deck::add_bottom(const Card &new_card)
{
    if (is_full())
        return 0;

    for (int i = nr_cards-1; i >= 0; i--)
        ptr_cards[i+1] = ptr_cards[i];
    
    ptr_cards[0] = new_card;
    nr_cards++;
    return 1;
}


/*
    Goal: remove the card on top of the deck (at location nr_cards-1)
    Inputs: none
    Outputs: A card object
    Processing: 
        - decrement nr_cards
        - return the card object at position nr_cards-1
    Preconditions: deck is not empty (nr_cards > 0)
    Postconditions: deck has one less card
*/
Card Deck::remove_top()
{
    assert(nr_cards > 0);
    nr_cards--; // decrement the number of cards
    return ptr_cards[nr_cards]; // the previous top card
}

/*
    Goal: remove a card from a particular position in the deck
    Inputs: position of the card
    Outputs: a card object
    Processing: 
       - copy the card at position pos
       - move all cards from pos+1 to nr_cards-1 one position
        towards the beginning of the array
       -  decrement nr_cards
    Preconditions: pos >=0 and pos < nr_cards
    Postconditions:
*/
Card Deck::remove(int pos)
{
    assert(nr_cards>0);
    assert(pos >= 0 and pos < nr_cards);

    // make a copy of the card at index pos
    Card copy_card = ptr_cards[pos]; // make a copy of card at index pos (copy constructor is called)
    for (int i = pos+1; i < nr_cards; i++)
        ptr_cards[i-1] = ptr_cards[i];
    
    nr_cards--;
    return copy_card;
}

/*
    Goal: return the card at the top of the deck, but do not remove it
    Inputs: none
    Outputs: card object
    Processing:
        return card object at position nr_cards-1
    Preconditions: deck is not empty
    Postconditions: none = deck is not changed
*/
Card Deck::peek_top() const
{
    assert(nr_cards > 0);
    return ptr_cards[nr_cards-1];
}

/*
    Goal: return the number of cards in the deck
    Inputs: none
    Outputs: integer - nr_cards
    Processing: return nr_cards
    Preconditions: none
    Postconditions: none
*/
int Deck::get_size() const
{
    return nr_cards;
}

/*
    Goal: check if deck is full
    Inputs: none
    Outputs: true if deck is full, false otherwise
    Processing: check if nr_cards is less than max_size
*/
bool Deck::is_full() const
{
    return nr_cards == max_size;
}

/*
    Goal: check if deck is empty
    Inputs: none
    Outputs: true if nr_cards = 0, false othersize
*/
bool Deck::is_empty() const
{
    return nr_cards == 0;
}

/*
    Goal: sort the cards in the deck by value
    Inputs: none
    Outputs: none
    Processing: sort cards in the deck, from smallest 
                value to largest value
                Use any sorting algorithm
                Use the operators > to compare two cards
                
    Preconditions: none
    Postconditions: none
*/
void Deck::sort() 
{
    // bubble-sort
    int nr_swaps  = 1,
        nr_passes = 0;
    while (nr_swaps > 0 and nr_passes < nr_cards)
    {
        // pass from i = 0 to nr_cards - nr_passes
        nr_swaps = 0;
        nr_passes++;
        for (int i = 0; i < nr_cards - nr_passes; i++)
        {
            if (ptr_cards[i] > ptr_cards[i+1]) // operator > is called from class Card
            {
                nr_swaps++;
                Card temp      = ptr_cards[i];
                ptr_cards[i]   = ptr_cards[i+1];
                ptr_cards[i+1] = temp;
            }
        }
    }
}

/*
    Goal: return the index of the largest card of a certain suit
    Inputs: suit
    Outputs: position of the card, or -1 if that suit was not found
    Processing: search for max value of cards of a certain suit
    Preconditions: none
    Postconditions: none
*/
int Deck::best_card(string suit) const
{
    int max_face = -1,
        max_ind = -1;
    for (int i = 0; i < nr_cards;i++)
    {
        if (ptr_cards[i].get_suite() == suit)
            if (ptr_cards[i].get_face() > max_face)
            {
                max_face = ptr_cards[i].get_face();
                max_ind = i;
            }
    }
    return max_ind;
}

/*
    Goal: return the index of the worst card of a certain suit
    Inputs: suit
    Outputs: position of the card, or -1 if that suit was not found
    Processing: search for min face of cards of a certain suit
    Preconditions: none
    Postconditions: none
*/
int Deck::worst_card(string suit) const
{
     int min_face = 100,
         min_ind  = -1;
    for (int i = 0; i < nr_cards;i++)
    {
        if (ptr_cards[i].get_suite() == suit)
            if (ptr_cards[i].get_face() < min_face)
            {
                min_face = ptr_cards[i].get_face();
                min_ind = i;
            }
    }
    return min_ind;
}

/*
    Goal: return the index of a card with a particular face for any suit
    Inputs: face
    Outputs: position of the card, or -1 if that face was not found
    Processing: search a card with a face value = face 
    Preconditions: none
    Postconditions: none
*/
int Deck::match_face(int face) const
{   
    int ind_card = -1, 
               i = 0;

    while (ind_card == -1 && i < nr_cards)
    {
        if (ptr_cards[i].get_face() == face)
            ind_card = i;
        i++;
    }
    return ind_card;
}
/*
    Goal: display the cards in the deck
    Inputs: none
    Outputs: none
    Processing: for each card in the deck call 
                    display function in the Card class
*/
void Deck::display() const
{
   for (int c = 0; c < nr_cards; c++)
        cout << c << ". " << ptr_cards[c].display() << endl;
}

/*
    Goal: display a card from the deck at position pos
    Inputs: integer pos = position of the card
    Outputs: none
    Processing: 
            call display function in the Card class for the card
            at location pos
    Precondtion: pos is between 0 and nr_cards-1, deck is not empty
*/
void Deck::display(int pos) const
{
    assert(nr_cards > 0);
    assert(pos >= 0 && pos < nr_cards);
    cout << ptr_cards[pos].display() << endl;
}
