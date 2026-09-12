#include <string>
#include "Cards.h"
using namespace std;

#ifndef DECK
#define DECK

class Deck
{
   private: // data members
		Card *ptr_cards; // pointer to an array of cards
		int nr_cards;    // current size deck
		int max_size;   // max # of cards in the Deck
		
	public: // function members
		Deck(int new_max_size = 52); // constructor empty deck of max_size
		Deck(const Deck &other); // copy constructor
		~Deck(); // destructor - deallocates dynamic memory in the Deck object
		Deck & operator=(const Deck &other); // assignment operator
		
		int  init_deck(string suits[], int size_suits, int min_val, int max_val);
		void shuffle(int nr_shuffles = 100); // shuffle all cards in the deck
		int  add_top(const Card &);
      	int  add_bottom(const Card &);
      	Card remove_top();
		Card remove(int pos);
		void sort();
		Card peek_top() const;
      	int  get_size() const;
	    bool is_full() const;
      	bool is_empty() const;
		int  best_card(string suite) const;
		int  worst_card(string suite) const;
		int  match_face(int face) const; 
		void display() const;
		void display(int pos) const;
};	

#endif
