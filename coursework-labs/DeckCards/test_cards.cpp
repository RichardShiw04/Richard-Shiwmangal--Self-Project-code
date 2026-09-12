#include <iostream>
#include <string>
#include <cassert>
using namespace std;
#include "Cards.h"

int main()
{
    Card c1;
    Card c2(10, "diamond");
    cout << c2.display() << endl;
    
    cout << "\nSuite = black diamond is " << (c2.check_suite("black diamond") ? "Valid": "Invalid") << endl; 
    cout << "Suite = diamonds is " << (c2.check_suite("diamonds") ? "Valid": "Invalid") << endl; 

    Card c3(15, "red hearts");
    cout << c3.display() << endl;

    c3 = c2;
    cout << c3.display() << endl;
}