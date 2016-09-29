#include <iostream>
#include "EM_Class.cpp"
#include <vector>
#include <string>
#include<iostream>
#include<time.h>
//#include<omp.h>
#include<chrono>

using namespace std;

inline double rand01();

int main()
{
	//this is the latest one
	//EM_Class test(100,2.5,1,2.5,28,7);
	srand (time(NULL));
	size_t samplesize =10;
	size_t number_of_terms=300;
	vector<EM_Class> testvector(samplesize);
	size_t num_iterations =250;
	string Dataset_Loc ="R_real_sub_diff1.txt";
	EM_Class param_getter;
	param_getter.load_R(Dataset_Loc);
	double average = param_getter.average();
	cout << average << endl;
	double bipower_sigma_s = param_getter.bipower_sigma_s();
	cout << bipower_sigma_s <<endl;
	double lambda_start;
	double tau_s_start = param_getter.tau_estim(average,bipower_sigma_s,lambda_start);
	cout << "tau start is " << tau_s_start <<" lambda start is " << lambda_start<< endl;	
//	string output = "output100_2.txt";
//	testvector.at(0).set_output(output);
//	EM_Class::set_output(output);
	ofstream output1;
	//output1.open("papertest_1.txt");
	output1.open("Real1_ResultsBrownianTest.txt");
	if(lambda_start<0)
	{
		lambda_start =rand01()/10.0;
	}
	if(tau_s_start<0)
	{
		tau_s_start = (1.5 - rand01())*bipower_sigma_s;
		lambda_start =rand01()/5.0;
	}

    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

	for(size_t i =0;i<samplesize;++i)
	{
/*		double lambda= rand01(); // INITIALISING INITIAL POSITIONS
		double mu = rand01()/1000;
		double nu= rand01()/1000;
		double sigma_s= 0.0001871595;//rand01(); // 9.313379e-11;// rand01();//1000.0;;
		double tao_s= rand01()/1000;//2.0e-10;    //rand01(); */
		double lambda= lambda_start*(1.5-rand01()); // INITIALISING INITIAL POSITIONS
		double mu = average*(1.5-rand01());
		double nu= 2*average*(0.5 - rand01());
		double sigma_s= bipower_sigma_s*(1.5-rand01()); // 9.313379e-11;// rand01();//1000.0;;
//		double sigma_s= bipower_sigma_s;
		double tao_s= tau_s_start*(1.5 -rand01());   //rand01();

		testvector.at(i).load(number_of_terms,mu,nu,lambda,tao_s,sigma_s);
		testvector.at(i).load_R(Dataset_Loc);
	//	testvector.at(i).load_R("R_real_sub_diff3.txt");
		testvector.at(i).Expectation_Maximization(num_iterations);
    //	cout << mu <<" "<< nu <<" "<< lambda <<" "<< sigma_s <<" " << tao_s <<" "<< endl;
		output1 << mu <<" "<< nu <<" "<< lambda <<" "<< sigma_s <<" " << tao_s <<" ";
		testvector.at(i).print_out_stream(output1);
	}

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count()/pow(10,6);
    cout << "The time to convergence was " << duration << endl;
	
	output1.close();
	return 0;
}

inline double rand01()
{
	return double(rand())/double(RAND_MAX);
}
