//#include <gsl/gsl_randist.h>
#define RCPP_FLAG 1
#include <iostream>
#include"EM_Class.h"
#include<cmath>
#include<chrono>
#include<time.h>


#if RCPP_FLAG==1
  #include<Rcpp.h>
#endif

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

void EM_Class::set_R(std::vector<double> R_in){

  //  R=R_in;
    full_returns=R_in;
    R.assign(full_returns.end() - std::min(full_returns.size(), max_window_size_ ), full_returns.end());

}

inline double EM_Class::normal_pdf(double R_n,int k)
{
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double s = sqrt(initial.sigma_s + k*initial.tau_s);
    double a = (R_n-initial.mu - k*initial.nu) / s;
    double temp = (inv_sqrt_2pi / (s) )* exp(-0.5 * a * a);
 //   static size_t i =0;

    /*   if(std::isnan(temp)&& i<10)    {
	 ++i;
	 std::cout << initial.sigma_s << " " << initial.tau_s << std::endl;
	 std::cout << sqrt(initial.sigma_s + k*(initial.tau_s)) << std::endl;
	 std::cout <<s << " " << a << " " << temp << std::endl;
	 }*/

    return temp;
}

inline double EM_Class::normal_pdf_no_const(double R_n,int k)
{
    double s = sqrt(initial.sigma_s + k*initial.tau_s);
    double a = (R_n-initial.mu - k*initial.nu) / s;
    double temp = exp(-0.5 * a * a)/s;

    return temp;
}
/*
double EM_Class::compute_new_return(double K)
{
  if(price_flag_)
  {
    //USE S vector
    //figure out times and infer the new return
    return K;
  }
  else
  {
    return K;
  }
}
*/

std::vector<double> EM_Class::update_return_singular(double K)
{
  int T = R.size();
  full_returns.push_back(K);

  parameters old_param=initial;
  parameters new_param=initial;

  //Need to change the returns set up, create R from underlying full returns vector

  //parameters saved_param=initial;

  //double new_return=K;//WORK OUT NEW RETURN

  new_param= Expectation_Maximization_OneStep_OneData(K,new_param);

  //IS THIS DATA INSIDE THE WINDOW
 /* if(T< max_window_size_)
  {
    //old_param= Expectation_Maximization_OneStep_OneData(R.at(i),old_param);
    initial.mu = (initial.mu*T +new_param.mu)/(T+1);
    initial.nu = (initial.nu*T +new_param.nu)/(T+1);
    initial.lambda = (initial.lambda*T+new_param.lambda)/(T+1);
    initial.sigma_s = (initial.sigma_s*T +new_param.sigma_s)/(T+1);
    initial.tau_s = (initial.tau_s*T +new_param.tau_s)/(T+1);
  }
  else
  {
    old_param= Expectation_Maximization_OneStep_OneData(R.at(0),old_param);
    initial.mu = (initial.mu*T - old_param.mu+new_param.mu)/T;
    initial.nu = (initial.nu*T - old_param.nu+new_param.nu)/T;
    initial.lambda = (initial.lambda*T - old_param.lambda+new_param.lambda)/T;
    initial.sigma_s = (initial.sigma_s*T - old_param.sigma_s+new_param.sigma_s)/T;
    initial.tau_s = (initial.tau_s*T - old_param.tau_s+new_param.tau_s)/T;
  }
*/
 //ADD NEW INFO TO DATA
//  R.at(i)=K;



  update_return_vec();

  if(max_em_iterations_>0)
  {
    Expectation_Maximization();
  }
  return initial.get_params();
}

void EM_Class::update_return_vec()
{
  R.assign(full_returns.end() - std::min(full_returns.size(), max_window_size_ ), full_returns.end());
}


parameters EM_Class::Expectation_Maximization_OneStep_OneData(double R_n,parameters starting){

  size_t T = R.size();
 // size_t i =0;
  //omp_set_num_threads(num_threads_);
  parameters prev_params =initial;
 // start_=starting;

 //   ++i;
    double beta_s = starting.tau_s/starting.sigma_s;
    double aver =  starting.mu  - (starting.nu/beta_s);
    prev_params=starting;
    double nu_sum=0;
    double mu_sum=0;
    double lambda_sum=0;
    double sigma_sum =0;
    double tau_term=0;

//#pragma omp parallel for reduction(+:mu_sum,nu_sum,lambda_sum,sigma_sum,tau_term)
  //  size_t  n=0;
      double a_n,b_n;
      double lambda_n = generate_lambda_an_bn_2(a_n,b_n,R_n);
      double temp = aver + a_n*(R_n - aver);
      double temp2 = (1.0 -a_n)*(R_n - aver);
      double c_n = beta_s*a_n*(1.0-a_n)- beta_s*b_n -((1.0-a_n)*(1.0-a_n))/lambda_n;

      lambda_sum+=lambda_n;
      mu_sum += temp;
      nu_sum += temp2;
      sigma_sum += starting.sigma_s*(1.0-a_n) + b_n*(R_n - aver)*(R_n - aver) + temp*temp;
      tau_term += starting.tau_s*(lambda_n -1 + a_n) + c_n*(R_n -aver)*(R_n -aver) + temp2*temp2/lambda_n;


    starting.lambda = lambda_sum/double(T);
    starting.nu = nu_sum/lambda_sum;
    starting.mu = mu_sum/double(T);
    starting.sigma_s = (sigma_sum/double(T)) - starting.mu*starting.mu;
    starting.tau_s = (tau_term/lambda_sum) -starting.nu*starting.nu;


  return starting;
  //FIX THIS, NEED TO RETURN CONVERGENCE INFO.
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

    double start_tau_s=(b-(1+a)*(1+a))*bipower_sigma_s;
    //                          //  cout << "this is tau " << start_tau_s << endl;
    return start_tau_s;

}

void EM_Class::load(size_t a,double mu1,double nu1,double s,double t,double l)
{
    max_poisson_terms_=a;
    initial.lambda = l;
    initial.nu = nu1;
    initial.mu = mu1;
    initial.tau_s = t;
    initial.sigma_s =s;
}

void EM_Class::load_params(double mu1,double nu1,double s,double t,double l)
{
    initial.lambda = l;
    initial.nu = nu1;
    initial.mu = mu1;
    initial.tau_s = t;
    initial.sigma_s =s;
}
void EM_Class::set_max_poisson_terms(size_t max_poisson_terms)
{
    max_poisson_terms_=max_poisson_terms;
}

void EM_Class::print_out_stream(ofstream &output1)
{
    output1 << initial.mu << " " << initial.nu << " " << initial.lambda << " " << initial.sigma_s << " " << initial.tau_s << " " << iterations_completed_ << " " << final_log_likelihood <<  " " << incomp_log_likelihood() << endl;

}
double EM_Class::incomp_log_likelihood()
{
    return incomp_log_likelihood_threaded_clean();
}




/*
double EM_Class::incomp_log_likelihood_threaded(){
    double incomplete_log_likelihood=0;
    size_t T=R.size();
#pragma omp parallel for reduction(+:incomplete_log_likelihood )
    for(size_t n=0; n<T; ++n)
    {
	double product =1;
	double exptemp = exp(-initial.lambda);
	double temp1 = exptemp*product*exp(-(R.at(n)-initial.mu)*(R.at(n)-initial.mu)/(2*(initial.sigma_s)))/sqrt(initial.sigma_s);
	double templik = (temp1);
	for(int k=1; k<30; ++k)
	{
	    product *=(initial.lambda/double(k));
	    templik += exptemp*product*exp(-(R.at(n)-initial.mu - k*initial.nu)*(R.at(n)-initial.mu - k*initial.nu)/(2*(initial.sigma_s +k*initial.tau_s)))/sqrt(initial.sigma_s +k*initial.tau_s);
	}
	incomplete_log_likelihood += log(templik/sqrt(2.0*3.14159265359));

    }
    return incomplete_log_likelihood;
}
*/
double EM_Class::incomp_log_likelihood_threaded_clean(){
    double incomplete_log_likelihood=0;
    size_t T=R.size();
#pragma omp parallel for reduction(+:incomplete_log_likelihood)
    for(size_t n=0; n<T; ++n)
    {
	double product =1;
	double exptemp = std::exp(-initial.lambda);
	double temp1 = exptemp*product*normal_pdf_no_const(R.at(n),0);
	double temp_lik = (temp1);
	double prev_temp_lik;
	size_t k =1;
	do
	{
	    prev_temp_lik = temp_lik;
	    product *=(initial.lambda/double(k));
	    temp_lik += exptemp*product*normal_pdf_no_const(R.at(n),k);
	    ++k;
	} while(prev_temp_lik!=temp_lik && k<max_poisson_terms_)	;
	//cout << k << endl;
	incomplete_log_likelihood += std::log(temp_lik/std::sqrt(2.0*3.14159265359));

    }
    return incomplete_log_likelihood;
}


/*
double EM_Class::incomp_log_likelihood_threaded_2(){
    double incomplete_log_likelihood=0;
    size_t T=R.size();
#pragma omp parallel for reduction(+:incomplete_log_likelihood )
    for(size_t n=0; n<T; ++n)
    {
	double product =1;
	double exptemp = exp(-initial.lambda);
	double temp1 = exptemp*product*normal_pdf_no_const(R.at(n),0);
	//double temp1 = exptemp*product*exp(-(R.at(n)-initial.mu)*(R.at(n)-initial.mu)/(2*(initial.sigma_s)))/sqrt(initial.sigma_s);
	double templik = (temp1);
	for(int k=1; k<50; ++k)
	{
	    product *=(initial.lambda/double(k));
	    templik += exptemp*product*normal_pdf_no_const(R.at(n),k);
	}
	incomplete_log_likelihood += log(templik/sqrt(2.0*3.14159265359));

    }
    return incomplete_log_likelihood;
}

*/

double EM_Class::generate_lambda_an_bn(double &a_n,double &b_n,double R_n)
{
    double beta_s = (initial.tau_s)/(initial.sigma_s);
    double product =1;
    double temp = normal_pdf(R_n,0);
    double sum_l_nom = 0;
    double sum_a_nom = temp;
    double sum_c_nom = temp;
    double sum_denom = temp;

    for(size_t k=1;k<max_poisson_terms_;++k)
    {
	product *= initial.lambda/k;
	//temp =gsl_ran_gaussian_pdf (R_n-initial.mu - k*initial.nu,sqrt(initial.sigma_s + k*initial.tau_s))*product;
	temp =normal_pdf(R_n,k)*product;
	//std::cout << product << std::endl;


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
    while(prev_sum_l_nom!=sum_l_nom && k<max_poisson_terms_)
    {
        prev_sum_l_nom = sum_l_nom;
        product *= initial.lambda/k;
        //temp =gsl_ran_gaussian_pdf (R_n-initial.mu - k*initial.nu,sqrt(initial.sigma_s + k*initial.tau_s))*product;
        temp =normal_pdf(R_n,k)*product;

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

double EM_Class::generate_lambda_an_bn_2_vec(double &a_n,double &b_n,const double R_n)
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
    const size_t vec_size=4;

    size_t k =1;
    while(prev_sum_l_nom!=sum_l_nom && k<max_poisson_terms_)
    {
	double sub_l_nom[4];
	double sub_a_nom[4];
	double sub_c_nom[4];
	double sub_denom[4];
	double sub_product[4];
	sub_product[0]=initial.lambda/k;
	sub_product[1]=initial.lambda*initial.lambda/(k*k+1);
	sub_product[2]=initial.lambda*initial.lambda*initial.lambda/(k*(k+1)*(k+2));
	sub_product[3]=initial.lambda*initial.lambda*initial.lambda*initial.lambda/(k*(k+1)*(k+2)*(k+3));

        prev_sum_l_nom = sum_l_nom;

	for(size_t j=0;j<vec_size;++j)
	{
	    temp =normal_pdf(R_n,k+j)*product;
	    double g_N = (1.0/(1.0+(k+j)*beta_s));
	    sub_l_nom[j]= (k+j)*temp;
	    sub_a_nom[j]= g_N*temp;
	    sub_c_nom[j]= g_N*g_N*temp;
	    sub_denom[j]= temp;
	    //sub_product[j]=initial.lambda/(k+j);
	}
	//for(size_t j=1;j<vec_size;++j){
	  //  sub_product[j]*=sub_product[j-1];
	//}
	for(size_t j=0;j<vec_size;++j)
	{
	    sum_l_nom+= sub_l_nom[j]*sub_product[j];
	    sum_a_nom+= sub_a_nom[j]*sub_product[j];
	    sum_c_nom+= sub_c_nom[j]*sub_product[j];
	    sum_denom+= sub_denom[j]*sub_product[j];

	}
        product*=sub_product[vec_size-1];



//        sum_l_nom+= k*temp;
  //      sum_a_nom+= g_N*temp;
    //    sum_c_nom+= g_N*g_N*temp;
     //   sum_denom+= temp;
        k+=vec_size;
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
std::vector<double> EM_Class::Expectation_Maximization(){

    size_t T = R.size();
    size_t i =0;
    //omp_set_num_threads(num_threads_);
    parameters prev_params =initial;
    start_=initial;

    do
    {
        ++i;
        double beta_s = initial.tau_s/initial.sigma_s;
        double aver =  initial.mu  - (initial.nu/beta_s);
        prev_params=initial;
        double nu_sum=0;
        double mu_sum=0;
        double lambda_sum=0;
        double sigma_sum =0;
        double tau_term=0;

#pragma omp parallel for reduction(+:mu_sum,nu_sum,lambda_sum,sigma_sum,tau_term)
        for(size_t n =0;n<T;++n)
        {
            double a_n,b_n;
            double R_n =R.at(n);
            double lambda_n = generate_lambda_an_bn_2(a_n,b_n,R_n);
            double temp = aver + a_n*(R_n - aver);
            double temp2 = (1.0 -a_n)*(R_n - aver);
            double c_n = beta_s*a_n*(1.0-a_n)- beta_s*b_n -((1.0-a_n)*(1.0-a_n))/lambda_n;

            lambda_sum+=lambda_n;
            mu_sum += temp;
            nu_sum += temp2;
            sigma_sum += initial.sigma_s*(1.0-a_n) + b_n*(R_n - aver)*(R_n - aver) + temp*temp;
            tau_term += initial.tau_s*(lambda_n -1 + a_n) + c_n*(R_n -aver)*(R_n -aver) + temp2*temp2/lambda_n;

        }
        initial.lambda = lambda_sum/double(T);
        initial.nu = nu_sum/lambda_sum;
        initial.mu = mu_sum/double(T);
        initial.sigma_s = (sigma_sum/double(T)) - initial.mu*initial.mu;
        initial.tau_s = (tau_term/lambda_sum) -initial.nu*initial.nu;

        if(debug_level_>0){
          //  double likelihood = incomp_log_likelihood();
           // std::cout << initial.mu << " " << initial.nu << " " <<  initial.sigma_s << " " << initial.tau_s << " "<< initial.lambda<<" "<<likelihood << endl;
            print_params();
        }
        if(debug_level_>2){
            print_likelihood();
        }
    }while(!convergence_test(i,prev_params));
    //double likelihood = incomp_log_likelihood();

    if(debug_level_>-1){
    //    std::cout << initial.mu << " " << initial.nu << " " <<  initial.sigma_s << " " << initial.tau_s<< " "<< initial.lambda<< " " << endl;
        print_params();
       // double likelihood = incomp_log_likelihood();
    }   //NEED BETTER CONVERGENCE CRITERIONS

    return initial.get_params();
    //FIX THIS, NEED TO RETURN CONVERGENCE INFO.
}

bool EM_Class::convergence_test(size_t i,parameters prev_params)
{
    double distance_sqr = eucl_dist_sqr(prev_params,initial);
    double rel_distance_sqr = rel_eucl_dist_sqr(prev_params,initial);
    if(debug_level_>1)
    {
     #if RCPP_FLAG==1
        Rcpp::Rcout << "Stopping Check, iteration number = "<< i << " ,distance_sqr = " << distance_sqr <<" ,rel_distance_sqr = " << rel_distance_sqr << endl;
     #else
        cout << "Stopping Check, iteration number = "<< i << " ,distance_sqr = " << distance_sqr <<" ,rel_distance_sqr = " << rel_distance_sqr << endl;
     #endif
    }
    //if(i<max_em_iterations_&&max_iteration_stopping_)
    bool dist_converg=distance_convergence(distance_sqr,rel_distance_sqr);
    if((i>=max_em_iterations_)||dist_converg)
    {
        final_rel_distance_s_=rel_distance_sqr;
        final_rel_distance_s_=rel_distance_sqr;
        iterations_completed_=i;
        return true;
    }
    else
    {
        return false;
    }
}

bool EM_Class::distance_convergence(double distance_sqr,double rel_distance_sqr)
{
    bool rel_d=false;
    bool d=false;
    if(rel_distance_flag_&&rel_distance_sqr<rel_distance_s_tol_)
    {
       // cout << "IN HERE HEREHREHREHRERHREHREHRERHERHERHERHERHERHERHERHERHERHERHERHERHER" << endl;
        rel_d=true;
    }
    if(distance_flag_&&distance_sqr<distance_s_tol_)
    {
        d=true;
    }
    return (rel_d||d);
}

double EM_Class::eucl_dist_sqr(parameters prev_params, parameters current_params)
{
    double distance =0.0;
    double x =(current_params.mu-prev_params.mu);
    distance += x*x;
    x =(current_params.nu-prev_params.nu);
    distance += x*x;
    x =(current_params.lambda-prev_params.lambda);
    distance += x*x;
    x =(current_params.sigma_s-prev_params.sigma_s);
    distance += x*x;
    x =(current_params.tau_s-prev_params.tau_s);
    distance += x*x;
    return distance;
}

double EM_Class::rel_eucl_dist_sqr(parameters prev_params, parameters current_params)
{
    double distance =0.0;
    double x =(current_params.mu-prev_params.mu)/prev_params.mu;
    distance += x*x;
    x =(current_params.nu-prev_params.nu)/prev_params.nu;
    distance += x*x;
    x =(current_params.lambda-prev_params.lambda)/prev_params.lambda;
    distance += x*x;
    x =(current_params.sigma_s-prev_params.sigma_s)/prev_params.sigma_s;
    distance += x*x;
    x =(current_params.tau_s-prev_params.tau_s)/prev_params.tau_s;
    distance += x*x;
    return distance;
}

EM_Class::EM_Class(size_t a,double mu1,double nu1,double l,double t,double s)
{
    max_poisson_terms_=a;
    initial.lambda = l;
    initial.nu = nu1;
    initial.mu = mu1;
    initial.tau_s = t;
    initial.sigma_s =s;
    debug_level_=-1;
    rel_distance_s_tol_=1e-15;
    distance_s_tol_=1e-10;
    rel_distance_flag_=1;
    distance_flag_=0;
    num_threads_=1;
    max_em_iterations_=500;

}
EM_Class::EM_Class()
{
    debug_level_=-1;
    rel_distance_s_tol_=1e-15;
    distance_s_tol_=1e-10;
    rel_distance_flag_=1;
    distance_flag_=0;
    num_threads_=1;
    max_em_iterations_=500;
    max_poisson_terms_=100;
}

void EM_Class::set_debug_level(int i)
{
    debug_level_=i;
}

void EM_Class::set_rel_dist_conv_flag(bool flag)
{
    distance_flag_=flag;
}
void EM_Class::set_dist_conv_flag(bool flag)
{
    distance_flag_=flag;
}
void EM_Class::set_dist_s_tol(double tol)
{
    distance_s_tol_=tol;
}
void EM_Class::set_rel_dist_s_tol(double tol)
{
    rel_distance_s_tol_=tol;
}

void EM_Class::print_vec_to_file(vector<double> d)
{
    ofstream output1;
    output1.open("JumpNumberExpectation.txt");
    size_t n = d.size();
    for(size_t i=0;i<n;++i)
    {
        output1 << d.at(i) << endl;
    }
}

vector<double> EM_Class::expected_num_of_jumps()
{
  //  double incomplete_log_likelihood=0;
    size_t T=R.size();
    vector<double> num_jumps_expected(T);
#pragma omp parallel for
    for(size_t n=0; n<T; ++n)
    {
        double product =1;
        double exptemp = std::exp(-initial.lambda);
        double denom = exptemp*product*normal_pdf_no_const(R.at(n),0);
        double num=0.0;
        double prev_num;
        size_t k =1;
        do
        {
            prev_num = num;
            product *=(initial.lambda/double(k));
            double temp= exptemp*product*normal_pdf_no_const(R.at(n),k);
            num+=k*temp;
            denom+=temp;
            ++k;
        } while(prev_num!=num && k<max_poisson_terms_);
        //cout << k << endl;
        num_jumps_expected.at(n)=num/denom;
    }
 //   print_vec_to_file(num_jumps_expected);
    return num_jumps_expected;
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


vector<double> EM_Class::auto_EM(bool random)
{
    seed_random();
   // srand (time(NULL));
//    int n =R.size();
    double mean = average();
    cout <<" the average is" << average() << endl;
    double bipower_sigma_s_est= bipower_sigma_s();
    double lambda_start;
    double tau_s_start =tau_estim(mean,bipower_sigma_s_est,lambda_start);

  //  if(lambda_start<0)
  //  {
  //      lambda_start =rand01()/10.0;
  //  }
  //  if(tau_s_start<0)
  //  {
      cout <<"Methods of moments fails" << endl;
        tau_s_start = 2*(1.5 - rand01())*bipower_sigma_s_est;
        lambda_start =rand01()/10.0;
  //  }
 //   double zero =0;
    //Expectation_Maximization_2(nt, R,average,zero ,bipower_sigma_s,tau_s_start,lambda_start,core);
    double mu_start=mean;
    cout <<"mu start is" << mu_start << endl;
    double nu_start=0.0;
    double sigma_s_start=bipower_sigma_s_est;
    if(random)
    {
        mu_start*=(1.5-rand01());
        nu_start=2*mean*(0.5-rand01());
        sigma_s_start=bipower_sigma_s_est*(1.5-rand01());
        lambda_start*=(1.5-rand01());
        tau_s_start*=(1.5-rand01());
    }
    cout <<"mu start is" << mu_start << endl;
    //load(max_poisson_terms_,mu_start,nu_start,sigma_s_start,tau_s_start,lambda_start);
    load_params(mu_start,nu_start,sigma_s_start,tau_s_start,lambda_start);
    print_string("The starting conditions are ");
    print_params();
    Expectation_Maximization();

    return initial.get_params();
}

#if RCPP_FLAG==1
inline double EM_Class::rand01()
{
  double x =Rcpp::runif(1,0, 1)[0];
//  if(x>1||x<0||std::isnan(x))
//  {
//    Rcpp::Rcout << "RAND01 IS MISBEHAVING "<<x << endl;
//  }
  return x;
}
inline void EM_Class::print_params()
{
  Rcpp::Rcout << initial.mu << " " << initial.nu << " " <<  initial.sigma_s << " " << initial.tau_s << " "<< initial.lambda << endl;
}
inline void EM_Class::print_string(string x)
{
  Rcpp::Rcout << x;
}
inline void EM_Class::print_likelihood(){
  double likelihood = incomp_log_likelihood();
  //double likelihood2 = incomp_log_likelihood_threaded_clean();
  Rcpp::Rcout <<"The log likelihood is"<< likelihood << endl;
}
inline void EM_Class::print_double(double x){
  Rcpp::Rcout <<x << endl;
}
void EM_Class::end_timing(chrono::high_resolution_clock::time_point t1){
  chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
  double duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  Rcpp::Rcout << "The time to convergence was " << duration << endl;
}
inline void EM_Class::seed_random()
{
  //srand (time(NULL));
  Rcpp::Environment env("package:base");
  Rcpp::Function r_seed_setting= env["set.seed"];
  r_seed_setting(time(NULL));
}
#else
inline double EM_Class::rand01()
{
  return double(rand())/double(RAND_MAX);
}
void EM_Class::print_params()
{
  std::cout << initial.mu << " " << initial.nu << " " <<  initial.sigma_s << " " << initial.tau_s << " "<< initial.lambda << endl;
}
inline void EM_Class::print_likelihood(){
  double likelihood = incomp_log_likelihood();
  //double likelihood2 = incomp_log_likelihood_threaded_clean();
  cout <<"The log likelihood is"<< likelihood << endl;
}
inline void EM_Class::print_string(string x)
{
  std::cout << x;
}
inline void EM_Class::print_double(double x){
  cout <<x << endl;
}
void EM_Class::end_timing(chrono::high_resolution_clock::time_point t1){
  chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
  double duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  cout << "The time to convergence was " << duration << endl;
}
inline void EM_Class::seed_random()
{
  srand (time(NULL));
  //set.seed();
}
#endif



void EM_Class::set_max_iterations(size_t max_em_iterations)
{
    max_em_iterations_=max_em_iterations;

}

void EM_Class::set_thread_num(size_t n){

  omp_set_num_threads(n);
 // int a;
  }

void EM_Class::set_max_window_size(size_t max_window_size)
{
  max_window_size_=max_window_size;
}


std::vector<double> EM_Class::data_simulation(int n,std::vector<double> parameter_vector) {

    double mu =parameter_vector[0];
    double nu =parameter_vector[1];
    double sigma =sqrt(parameter_vector[2]);
    double tau =sqrt(parameter_vector[3]);
    double lambda =parameter_vector[4];

    std::vector<double> data(n);

    //std::default_random_engine de(time(0)); //seed
    //std::normal_distribution <double> nd(mu, sigma); //mean followed by stdiv

//    std::normal_distribution <double> jump(nu, tau); //mean followed by stdiv

  //  for(int i =0;i<n;++i)
    //{
    //  data.at(i) = nd(de);
    //}

    data=Rcpp::as<vector<double> >(Rcpp::rnorm(n,mu,sigma));



 //   std::default_random_engine generator;
//    std::poisson_distribution<int> distribution(lambda);

    std::vector<int> N(n);
  //  std::vector<double> jumpnorm(max_element(N);

    N=Rcpp::as<vector<int> >(Rcpp::rpois(n,lambda));

    for(int i=0; i<n; ++i)
    {
     // N.at(i) = distribution(generator);
     std::vector<double> JumpComp = Rcpp::as<vector<double> >(Rcpp::rnorm(N.at(i),nu,tau));
      for(int j =0; j < N.at(i);++j)
      {
        data.at(i)+= JumpComp.at(j);
      }
    }
    return data;
  }
/*
 std::vector<double> DataSimulation_comp(int n,std::vector<double> parameter_vector) {

double mu =parameter_vector[0];
double nu =parameter_vector[1];
double sigma =sqrt(parameter_vector[2]);
double tau =sqrt(parameter_vector[3]);
double lambda =parameter_vector[4];

std::vector<double> data(n);

std::default_random_engine de(time(0)); //seed
std::normal_distribution <double> nd(mu, sigma); //mean followed by stdiv

std::normal_distribution <double> jump(nu, tau); //mean followed by stdiv

for(int i =0;i<n;++i)
{
data.at(i) = nd(de);
}

std::default_random_engine generator;
std::poisson_distribution<int> distribution(lambda);

std::vector<int> N(n);

for(int i=0; i<n; ++i)
{
N.at(i) = distribution(generator);
for(int j =0; j < N.at(i);++j)
{
data.at(i)+= jump(de);
}
}
return data;
}
*/


#if RCPP_FLAG==1
RCPP_MODULE(mod_EM_Class) {
  Rcpp::class_<EM_Class>("EM_Class")
  .constructor()
  .constructor<int,double,double,double,double,double>()
  .method( "ExpectationMaximatization", &EM_Class::Expectation_Maximization )
  .method( "loadDataset",&EM_Class::set_R)
  .method( "autoEM",&EM_Class::auto_EM)
  .method( "setDebugLevel",&EM_Class::set_debug_level)
  .method("setMaxEMIterations",&EM_Class::set_max_iterations)
  .method("setMaxPoissonTerms",&EM_Class::set_max_poisson_terms)
  .method("expectedNumberOfJumps",&EM_Class::expected_num_of_jumps)
  .method("setThreadNumber",&EM_Class::set_thread_num)
  .method("dataSimulation",&EM_Class::data_simulation)
  .method("addNewReturn",&EM_Class::update_return_singular)
  .method("setMaxWindowSize",&EM_Class::set_max_window_size)
  //    .method( "Load_Dataset",&EM_Class::set_R)
  ;
}
#endif








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
    double temp1 = exptemp*product*exp(-(R.at(n)-initial.mu)*(R.at(n)-initial.mu)/(2*(initial.sigma_s)))/sqrt(initial.sigma_s);
    double templik = (temp1);
    for(int k=1;k<2;++k)
    {
        product *=(initial.lambda/double(k));
        templik += exptemp*product*exp(-(R.at(n)-initial.mu - k*initial.nu)*(R.at(n)-initial.mu - k*initial.nu)/(2*(initial.sigma_s +k*initial.tau_s)))/sqrt(initial.sigma_s +k*initial.tau_s);
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
   double temp1 = exptemp*product*exp(-(R.at(n)-initial.mu)*(R.at(n)-initial.mu)/(2*(initial.sigma_s)))/sqrt(initial.sigma_s);
   double templik = (temp1);
   for(size_t k=1;k<300;++k)
   {
   product *=(initial.lambda/double(k));
   templik += exptemp*product*exp(-(R.at(n)-initial.mu - k*initial.nu)*(R.at(n)-initial.mu - k*initial.nu)/(2*(initial.sigma_s +k*initial.tau_s)))/sqrt(initial.sigma_s +k*initial.tau_s);
   }
   incomplete_log_likelihood += log(templik/sqrt(2.0*3.14159265359));

   }
//      std::cout << "The incomplete log likelihood is " << incomplete_log_likelihood <<std::endl;
return incomplete_log_likelihood;
}
*/

