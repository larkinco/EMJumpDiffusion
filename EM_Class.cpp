//#include <gsl/gsl_randist.h>
#include <iostream>
#include"EM_Class.h"
#include<cmath>

using namespace std;

void EM_Class::load_R(std::string fileloc)
{
    std::ifstream data(fileloc.c_str());
    int counter =0;

    double val;
    while( data >> val )
    {
        R.push_back( val );

        //std::cout << ++counter << std::endl;
    }
    data.close();

}

void EM_Class::print_out_stream(ofstream &output1)
{
output1 << initial.mu << " " << initial.nu << " " << initial.lambda << " " << initial.sigma_s << " " << initial.tau_s << " " << iterations_completed << " " << final_log_likelihood <<  " " << incomp_log_likelihood1() << endl;

}
/*
double EM_Class::Generator(double &a_n,double &b_n,double R_n)
{
	double Z =

	for(int k=1;k<300;++k)
	{
		

	}
}


double EM_Class::pdf(double x, int k)
{  
	return exp( -1 * (x - initial.mu-k*initial.nu) * (x -  initial.mu-k*initial.nu) / (2 *(sigma_s +k*tau_s))) / (sqrt((sigma_s +k*tau_s)*2 * 3.14159265));
}
*/
void EM_Class::Expectation_Maximization3(int nt){

	int T = R.size();
	int i =0;


	while(i<nt)
	{
		++i;
		double beta_s = initial.tau_s/initial.sigma_s;
		double aver =  initial.mu  - (initial.nu/beta_s);
		double nu_sum=0;
		double mu_sum=0;
		double lambda_sum=0;
		double sigma_sum =0;
		double tau_term=0;

		#pragma omp parallel for reduction(+:mu_sum,nu_sum,lambda_sum,sigma_sum,tau_term)
		for(int n =0;n<T;++n)
		{
			double a_n;
			double R_n =R.at(n);
			double b_n;
			double lambda_n = generate_lambda_an_bn(a_n,b_n,R_n);
			lambda_sum+=lambda_n;

			double temp = aver + a_n*(R_n - aver);
			mu_sum += temp;
			double temp2 = (1.0 -a_n)*(R_n - aver);
			nu_sum += temp2;	
			
			sigma_sum += initial.sigma_s*(1.0-a_n) + b_n*(R_n - aver)*(R_n - aver) + temp*temp;

			double c_n = beta_s*a_n*(1.0-a_n)- beta_s*b_n -((1.0-a_n)*(1.0-a_n))/lambda_n;

			tau_term += initial.tau_s*(lambda_n -1 + a_n) + c_n*(R_n -aver)*(R_n -aver) + temp2*temp2/lambda_n;

		}
		initial.lambda = lambda_sum/double(T);
		initial.nu = nu_sum/lambda_sum;
		initial.mu = mu_sum/double(T);
		initial.sigma_s = (sigma_sum/double(T)) - initial.mu*initial.mu;
		initial.tau_s = (tau_term/lambda_sum) -initial.nu*initial.nu;
		std::cout << initial.mu << " " << initial.nu << " " <<  initial.sigma_s << " " << initial.lambda << " "<< initial.tau_s<< " " << beta_s << endl;	
//		double likelihood = incomp_log_likelihood1();
//		double likelihood2 = incomp_log_likelihood2();
		
	}

}


void EM_Class::Expectation_Maximization2(int nt)
{
    int T = R.size();
    std::vector <double> a(T);
    std::vector <double> b(T);
    std::vector <double> lambda_n(T);
    std::vector <double> mu_n(T);
    double log_likelihood =0;
    double last_log_likelihood =0.00000001;
    double distance =1;
    double distance_like=1;
    //for(int i=0;i<nt;++i)
    int i =0;

    //while(i<nt)
    while((distance > 1.0e-9)&&i<nt)
    //while((distance_like > 1.0e-14)&&i<nt)
    {
    	++i;
        double lambda_sum=0;
        double mu_sum=0;
        //double mu_sqr_sum=0;
        double nu_sum=0;
        double beta_s = (initial.tau_s)/(initial.sigma_s);
        double aver = initial.mu -initial.nu/beta_s;
        log_likelihood=0;
//	T =5;
  //      #pragma omp parallel for reduction(+:lambda_sum,mu_sum,nu_sum,log_likelihood)
	for(int n=0;n<T;++n)
        {
            double R_n = R.at(n);
            lambda_n.at(n) =generate_lambda_an_bn(a.at(n),b.at(n),R_n);
            lambda_sum += lambda_n.at(n);
//       	 	cout << lambda_n.at(n) << endl;
            mu_sum += (R_n -aver)*a.at(n);
         
	    nu_sum +=  (1.0-a.at(n))*(R_n-aver);
	    log_likelihood+= (R_n-initial.mu)*(R_n-initial.mu);
        }
	log_likelihood = -1.0/(2.0*initial.sigma_s)*log_likelihood - (double(T)/2)*log(2*3.14159265359*initial.sigma_s);
	double estim_lambda = lambda_sum/T;
        double estim_mu = (mu_sum/T) + aver;
        double estim_nu = (nu_sum/lambda_sum);//implicitly includes T division
	//std::cout << estim_nu << std::endl;
//        std::cout << estim_lambda << " " << estim_nu << " " << estim_mu << std::endl;

        double sigma_s_sum=0;

//	#pragma omp parallel for reduction(+:sigma_s_sum)
        for(int n=0;n<T;++n)
        {
            sigma_s_sum +=(aver + a.at(n)*(R.at(n)-aver)-estim_mu)*(aver + a.at(n)*(R.at(n)-aver)-estim_mu) + initial.sigma_s*(1.0-a.at(n)) + b.at(n)*(R.at(n)-aver)*(R.at(n)-aver);
            //std::cout << sigma_s_sum << std::endl;
        }
        double estim_sigma_s= sigma_s_sum/T;
	
      double tao_s_sum =0;

//	#pragma omp parallel for reduction(+:tao_s_sum)
        for(int n=0;n<T;++n)
        {
            double c_n = beta_s*a.at(n)*(1-a.at(n))- beta_s*(b.at(n)) -(1-a.at(n))*(1-a.at(n))/lambda_n.at(n);
    tao_s_sum += initial.tau_s*(lambda_n.at(n) -1 + a.at(n)) + c_n*(R.at(n)-aver)*(R.at(n)-aver) + lambda_n.at(n)*(estim_nu - ((1-a.at(n))*(R.at(n)-aver)/lambda_n.at(n)))*(estim_nu - ((1-a.at(n))*(R.at(n)-aver)/lambda_n.at(n)));
        }
        double estim_tao_s = tao_s_sum/(T*estim_lambda);

std::cout << initial.mu << " " << initial.nu << " " <<  initial.sigma_s << " " << initial.lambda << " "<< initial.tau_s<< " " << beta_s <<  " " <<(log_likelihood)<< "\n";
        //std::cout << initial.tau_s << std::endl;
//double estim_tao_s = initial.tau_s;
//	cout << lambda_sum << endl;
	distance = (initial.lambda - estim_lambda)*(initial.lambda - estim_lambda)+ (initial.mu - estim_mu )*(initial.mu - estim_mu) +  (initial.nu - estim_nu)*(initial.nu - estim_nu) + (initial.sigma_s - estim_sigma_s)*(initial.sigma_s - estim_sigma_s) +  (initial.tau_s - estim_tao_s)*( initial.tau_s - estim_tao_s);
        if(i>1)
	{
		distance_like= abs((last_log_likelihood -log_likelihood)/last_log_likelihood);
	}
	else
	{
		distance_like =1;
	}
	last_log_likelihood = log_likelihood;
	initial.lambda = estim_lambda;
        initial.mu = estim_mu;
        initial.nu = estim_nu;
        initial.sigma_s = estim_sigma_s;
        initial.tau_s = estim_tao_s;
	 double incomplete_log_likelihood=0;
	//#pragma omp parallel for reduction(+:incomplete_log_likelihood)
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
//			cout << "this is temp " << temp << endl;

		}
		incomplete_log_likelihood += log(templik/sqrt(2.0*3.14159265359));
	}
	cout << incomplete_log_likelihood << endl;
	if(i>0)
	{
		distance = abs((final_log_likelihood-incomplete_log_likelihood)/ incomplete_log_likelihood);
	}
//	if(isnan(distance)){
//
//		distance =1;
//	}
	
	final_log_likelihood =  incomplete_log_likelihood;
    }
    log_likelihood=0;
  /*  for(int n=0;n<T;++n)
    {
	    double R_n = R.at(n);
	    log_likelihood+= (R_n-initial.mu)*(R_n-initial.mu);
    }
    log_likelihood = -1.0/(2.0*initial.sigma_s)*log_likelihood - (double(T)/2)*log(2*3.14159265359*initial.sigma_s);
 */ 
   // final_log_likelihood = ;
    iterations_completed =i;



//    std::ofstream output;
  //  output.open("data.txt"); 
   // outputstream << initial.mu << " " << initial.nu << " " << initial.lambda << " " << initial.sigma_s << " " << initial.tau_s << " " << i << " " << log_likelihood << std::endl;
//	output.close();
}



void EM_Class::Expectation_Maximization(int nt)
{
    int T = R.size();
    cout << T << endl;
    std::vector <double> a(T);
    std::vector <double> b(T);
    std::vector <double> lambda_n(T);
    std::vector <double> mu_n(T);
    double tempa,tempb;
     double incomplete_log_likelihood=0;
	double diff =1;
    //for(int i =0;i<10000;++i)
    double i =0;
    while((diff>1.0e-10)&&i<nt)
    {
    	++i;
        double lambda_sum=0;
        double mu_sum=0;
        double mu_sqr_sum=0;
        double nu_sum=0;
        double beta_s = (initial.tau_s)/(initial.sigma_s);
        double aver = initial.mu -initial.nu/beta_s;
        double estim_sigma_s=0;
        double t_term =0;
	double old_likelihood = incomplete_log_likelihood;
//       	double brownian_log_likelihood =0;
       // double estim_tao_s=0;
	 

        #pragma omp parallel for reduction(+:lambda_sum,mu_sum,nu_sum,t_term,estim_sigma_s)
	for(int n=0;n<T;++n)//Loop for means and lambda
        {
            double R_n = R.at(n);
           //lambda_n.at(n) =generate_lambda_an_bn(a.at(n),b.at(n),R_n);
	   double a_n=0;
	   double b_n =0;
	   double lambda_n =generate_lambda_an_bn(a_n,b_n,R_n);
//	   brownian_log_likelihood+= (R_n-initial.mu)*(R_n-initial.mu);

	   
	   //lambda_sum += lambda_n.at(n);
	   lambda_sum +=lambda_n;
          // cout << lambda_n << " " << endl; // <<n << " " << R_n << endl;
 	    //cout << lambda_n.at(n) << endl;
            if(!std::isnan(lambda_n))
	            {
	//	                    cout << "lambdgood" << endl;
//				                    break;
}

            //double temp = a.at(n)*(R_n -aver) +aver;//ca=hange later
	    double temp =a_n*(R_n-aver) +aver;

            mu_sum += temp;
            //nu_sum +=  (1-a.at(n))*(R_n-aver); // potiential to optimize
	    nu_sum +=  (1.0-a_n)*(R_n-aver);
//     		if(std::isnan(nu_sum)){
//			break;
//		}

//	cout << (1-a.at(n)) << endl; 
       	    double c_n = beta_s*a_n*(1.0-a_n)- beta_s*(b_n) -(1.0-a_n)*(1.0-a_n)/lambda_n;
//       double c_n = beta_s*a.at(n)*(1-a.at(n))- beta_s*(b.at(n)) -(1-a.at(n))*(1-a.at(n))/lambda_n.at(n);
//t_term +=  (beta_s*a.at(n)*(1-a.at(n))- beta_s*b.at(n));
		t_term += initial.tau_s*(lambda_n -1.0 + a_n) + c_n*(R_n -aver)*(R_n -aver) +(1.0-a_n)*(R_n-aver)*(1.0-a_n)*(R_n-aver)/lambda_n;
//t_term += initial.tau_s*(lambda_n.at(n) -1 + a.at(n)) + c_n*(R_n -aver)*(R_n -aver) +(1-a.at(n))*(R_n-aver)*(1-a.at(n))*(R_n-aver)/lambda_n.at(n);
estim_sigma_s += temp*temp +initial.sigma_s*(1.0-a_n) + b_n*(R_n - aver)*(R_n - aver); 
//       estim_sigma_s += temp*temp +initial.sigma_s*(1-a.at(n)) + b.at(n)*(R_n - aver)*(R_n - aver);
      	 }

//	brownian_log_likelihood = -1.0/(2.0*initial.sigma_s)*brownian_log_likelihood - (double(T)/2)*log(2*3.14159265359*initial.sigma_s);
	double estim_lambda = (lambda_sum/double(T));
        double estim_mu = (mu_sum/double(T));
        double estim_nu = (nu_sum/(lambda_sum));
        estim_sigma_s = (estim_sigma_s/double(T)) - estim_mu*estim_mu;
        double estim_tao_s= t_term/(estim_lambda*double(T)) - estim_nu*estim_nu*estim_lambda;

	std::cout << initial.mu << " " << initial.nu << " " <<  initial.sigma_s << " " << initial.lambda << " "<< initial.tau_s<< " " << beta_s<< " ";
       
        initial.lambda = estim_lambda;
        initial.mu = estim_mu;
        initial.nu = estim_nu;
        initial.sigma_s = estim_sigma_s;
        initial.tau_s = estim_tao_s;
//	std::cout << initial.mu << " " << initial.nu << " " <<  initial.sigma_s << " " << initial.lambda << " "<< initial.tau_s<< " " << beta_s <<  "\n";
        if(std::isnan(estim_lambda))
        {
		cout << "lambda" << endl;
		break;
       }
	if(std::isnan(estim_nu))
        {
	                cout << "nu" << endl;
			                break;
			
	}
	if(std::isnan(estim_mu))
	        {
		                cout << "mu" << endl;
				                break;
		}
if(std::isnan(estim_sigma_s) )
        {
	                cout << "sigma" << endl;
			                break;
	}

	if(std::isnan(estim_tao_s))
	        {
		                cout << "tao" << endl;
			                break;
						        }




	 if(initial.sigma_s<0)
	 {
		 //std::cout <<"CRAP" << std::endl;
			                 //n=T;
					 //i=n;
	 }

	 incomplete_log_likelihood=0;
	// cout << incomplete_log_likelihood << endl;
	 
	 #pragma omp parallel for reduction(+:incomplete_log_likelihood )
	 for(int n=0;n<T;++n)
	 {
		 //double templik=0;
		 double product =1;
		 double exptemp = exp(-initial.lambda);
		 double temp1 = exptemp*product*exp(-(R.at(n)-initial.mu)*(R.at(n)-initial.mu)/(0.5*(initial.sigma_s)))/sqrt(initial.sigma_s);
		 double templik = (temp1);
		 for(int k=1;k<2;++k)
		 {
			 product *=(initial.lambda/double(k));
			 templik += exptemp*product*exp(-(R.at(n)-initial.mu - k*initial.nu)*(R.at(n)-initial.mu - k*initial.nu)/(0.5*(initial.sigma_s +k*initial.tau_s)))/sqrt(initial.sigma_s +k*initial.tau_s);
		 }
		 incomplete_log_likelihood += log(templik/sqrt(2.0*3.14159265359));
	 	 
	 }
	 cout << incomplete_log_likelihood << "\n";//" " << brownian_log_likelihood  << "\n";
	diff = abs((incomplete_log_likelihood-old_likelihood)/old_likelihood);
	 if(std::isnan(diff)){

	                 diff =1;
			         }

//	 std::cout << initial.mu << " " << initial.nu << " " <<  initial.sigma_s << " " << initial.lambda << " "<< initial.tau_s<< "\n";
	 if(initial.tau_s <0)
	 {
		 //std::cout <<"CRAP2" << std::endl;
	 }
 	//cout << incomplete_log_likelihood << endl;
        //std::cout << i << std::endl;
    }
    //std::cout << initial.lambda << std::endl;
    //std::cout << initial.nu << std::endl;
    //std::cout << initial.mu << std::endl;
    //std::cout << initial.tau_s << std::endl;
    //std::cout << initial.sigma_s << std::endl;
}

double EM_Class::generate_lambda_an_bn(double &a_n,double &b_n,double R_n)
{
    double beta_s = (initial.tau_s)/(initial.sigma_s);
    double product =1;
    double temp = normal_pdf(R_n,0);
    double sum_l_nom = 0;
    double sum_a_nom = temp;
    double sum_c_nom = temp;
    double sum_denom = temp;

//    #pragma omp parallel for reduction(+:sum_l_nom,sum_a_nom,sum_c_nom,sum_denom)
    for(int k=1;k<m;++k)
    {
        product *= initial.lambda/k;
        //temp =gsl_ran_gaussian_pdf (R_n-initial.mu - k*initial.nu,sqrt(initial.sigma_s + k*initial.tau_s))*product;
        temp =normal_pdf(R_n,k)*product;
        //std::cout << product << std::endl;
        //temp = gaussian_dist(R_n,k)*product;

	 if(std::isnan(temp))
	 {
//		 cout << "tempnan "  << k<<   endl;
		 //                      break;
	 }
        double g_N = (1.0/(1.0+k*beta_s));
        sum_l_nom+= k*temp;
        sum_a_nom+= g_N*temp;
        sum_c_nom+= g_N*g_N*temp;
        sum_denom+= temp;

    }
    a_n = sum_a_nom/sum_denom;
    b_n = (sum_c_nom/sum_denom) - a_n*a_n;
    return(sum_l_nom/sum_denom);
}
/*void load(int a,double mu1,double nu1,double l,double t,double s)
{
	m=a;
	initial.lambda = l;
	initial.nu = nu1;
	initial.mu = mu1;
	initial.tau_s = t;
	initial.sigma_s =s;
}
*/

/*void EM_Class::create_prob_vector(std::vector <double> &prob_vector1,double R_n)
{
    double product =1;
    prob_vector1.at(0) = gaussian_dist(R_n,0);
    for(int k =1;k<m;++k)
    {
        product *= initial.lambda/k;
        prob_vector1.at(k) = gaussian_dist(R_n,k)*product;
    }
}*/

/*double EM_Class::gaussian_dist(double R_n,int k)
{
    double var = initial.sigma_s + k*initial.tau_s;
    double x = R_n-initial.mu - k*initial.nu;
    return (1/sqrt(2*PI*(var)))*exp(-(x*x)/(2*var));
}
*/
/*double EM_Class::normal_pdf(double R_n,int k)
{
    
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double s = sqrt(initial.sigma_s + k*initial.tau_s);
    double a = (R_n-initial.mu - k*initial.nu) / s;

    return (inv_sqrt_2pi / (s) )* exp(-0.5 * a * a);
}*/


//double EM_Class::log_likelyhood()
//{



//}
