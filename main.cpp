#include <iostream>
#include "EM_Class.h"
//#include<omp.h>
using namespace std;

int main()
{
    //this is the latest one
    //EM_Class test(100,2.5,1,2.5,28,7);


EM_Class test(300,0.0005,0,0.000005,0.00000001,0.00000000000001);
test.load_R("Routput.txt");
cout << test.R.size() << endl;

//cout << test.R.size() << endl;
    test.Expectation_Maximization2(10000);
    //ofstream output;
    //output.open("final_answer.txt");
    //output << test.parameters
    return 0;
}
