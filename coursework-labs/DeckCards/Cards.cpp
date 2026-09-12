     
#include "Cards.h"
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

string Card::ALL_SUITE[4] = {"hearts", "diamonds", "spade", "clubs"};

 // default constructor + specific constructor (default values)
Card::Card(int new_face, string new_suite)
{
    if (new_face >=1 && new_face <= 14)
        set_face(new_face);
    else
        face = 1; 
    if (check_suite(new_suite))
        set_suite(new_suite);
    else    
        suite = "hearts";
}

/*
    Goal: copy constructor: create a new Card object as a 
          copy of an existing object
    Inputs: reference to a constant other card object
    Outputs: none
    Processing: 
    Preconditions: none
    Postconditions: new card has the same face and suite as other card
*/
Card::Card(const Card &other)
{
    suite = other.suite;
    face  = other.face;
}

// call set_face: card1.set_face(10) - card1 = calling object
void Card::set_face(int new_face)
{
    assert(new_face >=1 && new_face <= 14);
    face = new_face;  // changes the face of the calling object
}

// checks if string new_suite is a valid suite
 bool Card::check_suite(string new_suite) const
 {
    int ind_suite = 0;
    while (ind_suite < 4 && new_suite != ALL_SUITE[ind_suite])
        ind_suite++;
    if (ind_suite == 4)
    {
        cout << new_suite << " is not a valid suite " << endl;
        return false;
    }
    return true;
 }

// precondition new_suite if a valid suite
void Card::set_suite(string new_suite)
{
    assert(check_suite(new_suite));
    suite = new_suite;
}

int Card:: get_face() const       // accessor
{
    return face;
}
string Card::get_suite() const  // accessor
{
    return suite;
}

string Card::display() const
{
    return "face ="+ to_string(face) + ", suite = " + suite;
}

// if (card1 == card2) -> card1.operator==(card2)
bool Card::operator==(const Card &other) const // const Card &other = pass a reference to a constant object
{
    // face of the calling object
    // other.face // face of the other object (input parameter)  
    if (face == other.face && suite == other.suite)
        return true;
    else
        return false;
}
        
//Call: card1 = card2  -> card1.operator=(card2)
Card& Card::operator=(const Card &other) 
{
    if (*this == other) // self assignment
        return *this; 
    face  =  other.face;
    suite =  other.suite;
    return *this;
}

// compares the faces only
bool Card::operator>(const Card &other) const
 {
    return (face > other.face);
 }

// checks if a Card has the same suite as anothe card
bool Card::same_suite(const Card &other) const
{
    return (suite == other.suite);
}