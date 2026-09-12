#include <string>
using namespace std;

#ifndef CARD
#define CARD
/* class definition */

class Card
{
    private: // data members
        int face; // 1 to 13
        string suite; // possible values: {"hearts", "diamonds", "spade", "clubs"}
    public: 
        static string ALL_SUITE[4];

        // default constructor + specific constructor (default values)
        Card(int new_face = 1, string new_suite = "hearts");
        // copy constructor
        Card(const Card &other);
        void set_face(int new_face);
        void set_suite(string new_suite);
        bool same_suite(const Card &other) const;
        bool check_suite(string new_suite) const;
        int  get_face() const;       // accessor
        string get_suite() const;  // accessor
        string display() const; // accessor -returns a string
        // if (card1 == card2) -> card1.operator==(card2)
        bool operator==(const Card &other) const; // const Card &other = pass a reference to a constant object
        Card& operator=(const Card &other); // card0 = card1 = card2
        bool operator>(const Card &other) const; 
};
#endif