#include <gsl/gsl_randist.h>
#include <iostream>
#include"EM_Class.h"
#include<cmath>
#include<chrono>

using namespace std;

void EM_Class::load_R(std::string fileloc)
{
    std::ifstream data(fileloc.c_str());
    //int counter =0;

    double val;
    while( data >> val )
    {
        R.push_back( val );

        //std::cout << ++counter << std::endl;
    }
    data.close();
}

inline double EM_Class::normal_pdf(double R_n,int k)
{
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double s = sqrt(initial.sigma_s + k*initial.tau_s);
    double a = (R_n-initial.mu - k*initial.nu) / s;
    double temp = (inv_sqrt_2pi / (s) )* exp(-0.5 * a * a);
    static size_t i =0;

/*   if(std::isnan(temp)&& i<10)    {
       ++i;
       std::cout << initial.sigma_s << " " << initial.tau_s << std::endl;
       std::cout << sqrt(initial.sigma_s + k*(initial.tau_s)) << std::endl;
       std::cout <<s << " " << a << " " << temp << std::endl;
    }*/

    return temp;
}


double EM_Class::average(){
    size_t n =R.size();
    double sum=0;
    for(size_t i=0;i<n;++i)
    {
        sum += R.at(i);
    }
    return (sum/n);
}
double EM_Class::bipower_sigma_s()
{
    size_t n =R.size();
    double bipower_sum=0;
    for(size_t i=1;i<n;++i)
    {
        bipower_sum += fabs(R.at(i)*R.at(i-1));
    }
    return bipower_sum/(n*(0.79788)*(0.79788));
}
double EM_Class::tau_estim(double average,double bipower_sigma_s,double &lambda)
{
    size_t n =R.size();
    double var=0;
    double var4=0;
    for(size_t i=0;i<n;++i)
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
    //  cout << "lambda start is " << lambda << endl;

    double start_tau_s=(b-(1+a)*(1+a))*bipower_sigma_s;
    //                          //  cout << "this is tau " << start_tau_s << endl;
    return start_tau_s;

}

void EM_Class::load(size_t a,double mu1,double nu1,double l,double t,double s)
{
    m=a;
    initial.lambda = l;
    initial.nu = nu1;
    initial.mu = mu1;
    initial.tau_s = t;
    initial.sigma_s =s;
}

void EM_Class::print_out_stream(ofstream &output1)
{
    output1 << initial.mu << " " << initial.nu << " " << initial.lambda << " " << initial.sigma_s << " " << initial.tau_s << " " << iterations_completed << " " << final_log_likelihood <<  " " << incomp_log_likelihood() << endl;

}
double EM_Class::incomp_log_likelihood()
{
    return incomp_log_likelihood_threaded();
}

double EM_Class::incomp_log_likelihood_threaded(){
    double incomplete_log_likelihood=0;
    size_t T=R.size();
#pragma omp parallel for reduction(+:incomplete_log_likelihood )
    for(size_t n=0; n<T; ++n)
    {
        double product =1;
        double exptemp = exp(-initial.lambda);
        double temp1 = exptemp*product*exp(-(R.at(n)-initial.mu)*(R.at(n)-initial.mu)/(0.5*(initial.sigma_s)))/sqrt(initial.sigma_s);
        double templik = (temp1);
        for(int k=1; k<10; ++k)
        {
            product *=(initial.lambda/double(k));
            templik += exptemp*product*exp(-(R.at(n)-initial.mu - k*initial.nu)*(R.at(n)-initial.mu - k*initial.nu)/(0.5*(initial.sigma_s +k*initial.tau_s)))/sqrt(initial.sigma_s +k*initial.tau_s);
        }
        incomplete_log_likelihood += log(templik/sqrt(2.0*3.14159265359));

    }
    return incomplete_log_likelihood;
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

    for(size_t k=1;k<m;++k)
    {
        product *= initial.lambda/k;
        temp =gsl_ran_gaussian_pdf (R_n-initial.mu - k*initial.nu,sqrt(initial.sigma_s + k*initial.tau_s))*product;
        //temp =normal_pdf(R_n,k)*product;
        //std::cout << product << std::endl;

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


double EM_Class::generate_lambda_an_bn_2(double &a_n,double &b_n,double R_n)
{
  double beta_s = (initial.tau_s)/(initial.sigma_s);
  double product =1;
//  double temp = normal_pdf_fewer(R_n-mu,sigma_s);
  double temp = normal_pdf(R_n,0);

  double sum_l_nom = 0;
  double sum_a_nom = temp;
  double sum_c_nom = temp;
  double sum_denom = temp;
  double prev_sum_l_nom=1;

  size_t k =1;
  while(prev_sum_l_nom!=sum_l_nom && k<m)
  {
      prev_sum_l_nom = sum_l_nom;
      product *= initial.lambda/k;
     // temp =normal_pdf_fewer(R_n-mu - k*nu,sigma_s + k*tau_s)*product;
      //temp =normal_pdf(R_n,k,sigma_s,tau_s,mu,nu)*product;
      temp =gsl_ran_gaussian_pdf (R_n-initial.mu - k*initial.nu,sqrt(initial.sigma_s + k*initial.tau_s))*product;

      double g_N = (1.0/(1.0+k*beta_s));
      sum_l_nom+= k*temp;
      sum_a_nom+= g_N*temp;
      sum_c_nom+= g_N*g_N*temp;
      sum_denom+= temp;
      ++k;
  }
  //  std::cout << k << std::endl;
  a_n = sum_a_nom/sum_denom;
  b_n = (sum_c_nom/sum_denom) - a_n*a_n;
  return(sum_l_nom/sum_denom);
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
void EM_Class::Expectation_Maximization(size_t nt){

    size_t T = R.size();
    size_t i =0;
    omp_set_num_threads(24);

    while(i<nt)   //NEED BETTER CONVERGENCE CRITERIONS
    {
        ++i;
        double beta_s = initial.tau_s/initial.sigma_s;
        double aver =  initial.mu  - (initial.nu/beta_s);
        double nu_sum=0;
        double mu_sum=0;
        double lambda_sum=0;
        double sigma_sum =0;
        double tau_term=0;

//#pragma omp parallel for default(shared) reduction(+:mu_sum,nu_sum,lambda_sum,sigma_sum,tau_term)
#pragma omp parallel for default(shared)  schedule(dynamic)
	for(size_t n =0;n<T;++n)
        {
            double a_n,b_n;
            double const R_n =R.at(n);
           // double R_n = 0.1;
	    double lambda_n = generate_lambda_an_bn_2(a_n,b_n,R_n);
            double temp = aver + a_n*(R_n - aver);
            double temp2 = (1.0 -a_n)*(R_n - aver);
            double c_n = beta_s*a_n*(1.0-a_n)- beta_s*b_n -((1.0-a_n)*(1.0-a_n))/lambda_n;
//            lambda_sum+=lambda_n;
  //          mu_sum += temp;
    //        nu_sum += temp2;	
      //      sigma_sum += initial.sigma_s*(1.0-a_n) + b_n*(R_n - aver)*(R_n - aver) + temp*temp;
        //    tau_term += initial.tau_s*(lambda_n -1 + a_n) + c_n*(R_n -aver)*(R_n -aver) + temp2*temp2/lambda_n;
        }
       /* initial.lambda = lambda_sum/double(T);
        initial.nu = nu_sum/lambda_sum;
        initial.mu = mu_sum/double(T);
        initial.sigma_s = (sigma_sum/double(T)) - initial.mu*initial.mu;
        initial.tau_s = (tau_term/lambda_sum) -initial.nu*initial.nu;
*/        initial.lambda = 0.01;
        initial.nu = 0.001;
        initial.mu = 0.001;
        initial.sigma_s = 0.001;
        initial.tau_s = 0.001;
  //    double likelihood = incomp_log_likelihood();
        //        std::cout << initial.mu << " " << initial.nu << " " <<  initial.sigma_s << " " << initial.lambda << " "<< initial.tau_s<< " " << beta_s<<" "<<likelihood << endl;	
        //		double likelihood = incomp_log_likelihood1();
        //		double likelihood2 = incomp_log_likelihood2();

    }
    std::cout << initial.mu << " " << initial.nu << " " <<  initial.sigma_s << " " << initial.lambda << " "<< initial.tau_s<< " " << endl;	

}
EM_Class::EM_Class(size_t a,double mu1,double nu1,double l,double t,double s)
{
    m=a;
    initial.lambda = l;
    initial.nu = nu1;
    initial.mu = mu1;
    initial.tau_s = t;
    initial.sigma_s =s;
}
EM_Class::EM_Class()
{

}
/*
   void EM_Class::set_output(std::string filedest)
   {
   outputstream.open("outputtest10.txt");
   }
   void EM_Class::close_output()
   {
   outputstream.close();
   }
   */

chrono::high_resolution_clock::time_point EM_Class::start_timing(){
    return chrono::high_resolution_clock::now();
}

void EM_Class::end_timing(chrono::high_resolution_clock::time_point t1){
    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
    cout << "The time to convergence was " << duration << endl;
}
/*
   double EM_Class::thread_start_timing()
   {
   return omp_get_wtime()   
   }
   void EM_Class::thread_end_timing(double t1)
   {
   double t2 = omp_get_wtime()
   double duration = t2-t1
   cout << "The time (thread) to convergence was " << duration << endl;   
   }
   */

/*
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
*/
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

/*
   double EM_Class::incomp_log_likelihood2()
   {
   size_t T =R.size();
   double incomplete_log_likelihood=0;
   for(size_t n=0;n<T;++n)
   {
   double product =1;
   double exptemp = exp(-initial.lambda);
   double temp1 = exptemp*product*exp(-(R.at(n)-initial.mu)*(R.at(n)-initial.mu)/(0.5*(initial.sigma_s)))/sqrt(initial.sigma_s);
   double templik = (temp1);
   for(size_t k=1;k<300;++k)
   {
   product *=(initial.lambda/double(k));
   templik += exptemp*product*exp(-(R.at(n)-initial.mu - k*initial.nu)*(R.at(n)-initial.mu - k*initial.nu)/(0.5*(initial.sigma_s +k*initial.tau_s)))/sqrt(initial.sigma_s +k*initial.tau_s);
   }
   incomplete_log_likelihood += log(templik/sqrt(2.0*3.14159265359));

   }
//      std::cout << "The incomplete log likelihood is " << incomplete_log_likelihood <<std::endl;
return incomplete_log_likelihood;
}
 */

