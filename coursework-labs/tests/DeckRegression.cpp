#include "Deck.h"
#include <cassert>
#include <stdexcept>
#include <iostream>
int main(){
 Deck d(1); d.shuffle();d.add_top(Card(3,"hearts"));d.shuffle();assert(d.get_size()==1);
 Deck copy=d;assert(copy.remove_top().get_face()==3);assert(d.get_size()==1);
 Deck small(1);std::string suits[]={"hearts"};bool rejected=false;
 try{small.init_deck(suits,1,1,13);}catch(const std::invalid_argument&){rejected=true;}
 assert(rejected);assert(small.get_size()==0);
 std::cout<<"Empty/single shuffle, copy and capacity checks passed.\n";
}
