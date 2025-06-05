#include <iostream>

using namespace std;


#define ALPHA_LPF1_WIDTH  		   4
const uint8_t initial_alpha_LPF1		= 0b1101;



int main(int argc, char ** argv)
{
    for (int i = 0; i < ALPHA_LPF1_WIDTH; ++i)
    {
        bool bit = ((initial_alpha_LPF1 >> (ALPHA_LPF1_WIDTH - i - 1)) & 1);
    }

    return 0;
}