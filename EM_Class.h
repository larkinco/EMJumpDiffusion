#ifndef EM_CLASS_H
#define EM_CLASS_H
#include<vector>
#include"parameters.h"
#include <cmath>
//#include<math.h>
#include <string>
#include <vector>
#include<fstream>
#include<iostream>
#include<omp.h>
#include<string>
using namespace std;

const double PI = atan(1.0)*4;

class EM_Class{

    public:
        EM_Class(int a,double mu1,double nu1,double l,double t,double s)
        {
            m=a;
            initial.lambda = l;
            initial.nu = nu1;
            initial.mu = mu1;
            initial.tau_s = t;
            initial.sigma_s =s;
        }
	EM_Class()
	{

	};
        void print_out_stream(std::ofstream &output1);	
	void load(int a,double mu1,double nu1,double l,double t,double s)
	{
		m=a;
		initial.lambda = l;
		initial.nu = nu1;
		initial.mu = mu1;
		initial.tau_s = t;
		initial.sigma_s =s;
	}
	double average()
	{
		int n =R.size();
		double sum=0;
		for(int i=0;i<n;++i)
		{
			sum += R.at(i);
		}
		return (sum/n);
	}
	double bipower_sigma_s()
	{
		int n =R.size();
		double bipower_sum=0;
		for(int i=1;i<n;++i)
		{
			bipower_sum += fabs(R.at(i)*R.at(i-1));
		}
		return bipower_sum/(n*(0.79788)*(0.79788));
	}
	double tau_estim(double average,double bipower_sigma_s,double &lambda)
	{
		int n =R.size();
		double var=0;
		double var4=0;
		for(int i=0;i<n;++i)
		{
			double temp =(R.at(i)-average)*(R.at(i)-average);
			var +=temp;
			var4 +=temp*temp;
		}
		var = var/n;
		var4= var4/n;
		double a = (var/bipower_sigma_s)-1;
		double b = var4/(3*bipower_sigma_s*bipower_sigma_s);
		lambda = a/(b -(1+a)*(1+a));
	//	cout << "lambda start is " << lambda << endl;

		double start_tau_s=(b-(1+a)*(1+a))*bipower_sigma_s;
	//	cout << "this is tau " << start_tau_s << endl;
		return start_tau_s;
	
}
	static void set_output(std::string filedest)
	{
		//outputstream.open(filedest);
		outputstream.open("outputtest10.txt");
	}
	static void close_output()
	{
		outputstream.close();
	}
	void Expectation_Maximization3(int nt);												
        static std::ofstream outputstream;
	void load_R(std::string fileloc);
        void Expectation_Maximization(int nt);
        double gaussian_dist(double R_n,int k);
        //void create_prob_vector(std::vector <double> &prob_vector1,double R_n);
        double generate_lambda_an_bn(double &a_n,double &b_n,double R_n);
        void Expectation_Maximization2(int nt);
        //double normal_pdf(double R_n,int k);
        std::vector <double> R;
	inline double normal_pdf(double R_n,int k)
	{
	//	cout << initial.sigma_s << " " << initial.tau_s << endl;
		static const double inv_sqrt_2pi = 0.3989422804014327;
		double s = sqrt(initial.sigma_s + k*initial.tau_s);
		double a = (R_n-initial.mu - k*initial.nu) / s;
		double temp = (inv_sqrt_2pi / (s) )* exp(-0.5 * a * a);
		static int i =0;
	//	i++;
		
		if(std::isnan(temp)&& i<10)
		{
			++i;	
//			std::cout << initial.sigma_s << " " << initial.tau_s << std::endl;
//			std::cout << sqrt(initial.sigma_s + k*(initial.tau_s)) << std::endl;
//			std::cout <<s << " " << a << " " << temp << std::endl;
		}
		
		return temp;
	}

	double incomp_log_likelihood1()
	{
		int T =R.size();

		double incomplete_log_likelihood=0;
		//#pragma parallel omp for reduction(+:incomplete_log_likelihood)
		for(int n=0;n<T;++n)
		{
			//double templik=0;
			double product =1;
			double exptemp = exp(-initial.lambda);
			double temp1 = exptemp*product*exp(-(R.at(n)-initial.mu)*(R.at(n)-initial.mu)/(0.5*(initial.sigma_s)))/sqrt(initial.sigma_s);
			double templik = (temp1);
			for(int k=1;k<300;++k)
			{
				product *=(initial.lambda/double(k));
				templik += exptemp*product*exp(-(R.at(n)-initial.mu - k*initial.nu)*(R.at(n)-initial.mu - k*initial.nu)/(0.5*(initial.sigma_s +k*initial.tau_s)))/sqrt(initial.sigma_s +k*initial.tau_s);
				//                      cout << "this is temp " << temp << endl;

			}
			incomplete_log_likelihood += log(templik/sqrt(2.0*3.14159265359));
		}
//		std::cout << "The incomplete log likelihood is " << incomplete_log_likelihood << std::endl;
		return incomplete_log_likelihood;

	}
	double incomp_log_likelihood2()
	{
		int T =R.size();
		double incomplete_log_likelihood=0;
		for(int n=0;n<T;++n)
		{
			//double templik=0;
			double product =1;
			double exptemp = exp(-initial.lambda);
			double temp1 = exptemp*product*exp(-(R.at(n)-initial.mu)*(R.at(n)-initial.mu)/(0.5*(initial.sigma_s)))/sqrt(initial.sigma_s);
			double templik = (temp1);
			for(int k=1;k<300;++k)
			{
				product *=(initial.lambda/double(k));
				templik += exptemp*product*exp(-(R.at(n)-initial.mu - k*initial.nu)*(R.at(n)-initial.mu - k*initial.nu)/(0.5*(initial.sigma_s +k*initial.tau_s)))/sqrt(initial.sigma_s +k*initial.tau_s);
			}
			incomplete_log_likelihood += log(templik/sqrt(2.0*3.14159265359));

		}
//		std::cout << "The incomplete log likelihood is " << incomplete_log_likelihood <<std::endl;
		return incomplete_log_likelihood;
	}

    protected:
    private:
        parameters initial;
        parameters maximising;
        int m;
//        int n;
	double final_log_likelihood;
	int iterations_completed;
};

#endif // EM_CLASS_H
