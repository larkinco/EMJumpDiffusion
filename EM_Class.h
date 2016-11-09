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

#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_max_threads() { return 1;}
inline void omp_set_num_threads(int core){
  std::cout<<"Threading not enabled."<< std::endl;
}
#endif

#if RCPP_FLAG==1
#include<Rcpp.h>
#endif

#include<string>
//#include<chrono>
using namespace std;

//const double PI = atan(1.0)*4;

class EM_Class{

    public:
        EM_Class(size_t max_p_terms,double mu1,double nu1,double l,double t,double s);
        EM_Class();
        void load(size_t max_p_terms,double mu1,double nu1,double s,double t,double l);
        void load_params(double mu1,double nu1,double s,double t,double l);
        void set_max_poisson_terms(size_t max_poisson_terms);
        void print_out_stream(std::ofstream &output1);
        double average();
        double bipower_sigma_s();
        double tau_estim(double average,double bipower_sigma_s,double &lambda);
        //        static void set_output(std::string filedest);
        //      static void close_output();
        void set_R(std::vector<double> R_in);
        //    double thread_start_timing();
        //  void thread_end_timing(double t1);
        void load_R(std::string fileloc);
        std::vector<double> Expectation_Maximization();
        double incomp_log_likelihood();
        double incomp_log_likelihood_threaded_clean();
        vector<double> expected_num_of_jumps();
        void print_vec_to_file(vector<double> d);
        Rcpp::List auto_EM(bool random);
        void set_max_iterations(size_t max_em_iterations);
        void set_debug_level(int i);
        void set_rel_dist_conv_flag(bool flag);
        void set_dist_conv_flag(bool flag);
        void set_dist_s_tol(double tol);
        void set_rel_dist_s_tol(double tol);
        void set_thread_num(size_t n);
        //RETURN INFO ABOUT CONVERGENCE, NUMBER OF ITERATIONS, ERRORS
        void convergence_information();
        std::vector<double> Expectation_Maximization_modLikelihood();
        std::vector<double> data_simulation(int n,std::vector<double> parameter_vector);
        std::vector<double> update_return_singular(double K);
        std::vector<double> update_return_singular_2(double K);
        void set_max_window_size(size_t max_window_size);
        double prob_stock_price(double current_stock_price,double bound,int T);
        void print_convergence_info();
        std::vector<double> add_partial_return(double K,double fraction_of_timestep);
        void close_current_return();

//////////////DEBUG CODE//////////////////////////////////////
        vector<double> llikelihood_sequence;
        vector<double> rel_llikelihood_sequence;
        vector<double> param_dist_sequence;
        vector<double> rel_param_dist_sequence;
        vector<double> mu_seq;
        vector<double> nu_seq;
        vector<double> sigma_s_seq;
        vector<double> tau_s_seq;
        vector<double> lambda_seq;
////////////////////////////////////////////////////
    protected:
    private:
        std::vector <double> R;
        std::vector <double> full_returns;
        double partial_return_;
        bool prev_partial_return_;
        parameters initial;
        //SAVE STARTING CONDITIONS
        parameters start_;
        parameters maximising;
        size_t max_poisson_terms_;
        int debug_level_;
        //size_t m;
        size_t max_window_size_;
        double final_log_likelihood;
        size_t iterations_completed_;
        size_t max_em_iterations_;
        double final_rel_distance_s_;
        double final_distance_s_;
        double rel_distance_s_tol_;
        double distance_s_tol_;
        bool rel_distance_flag_;
        bool distance_flag_;
        double final_llikelihood_rel_dist_;
        double final_llikelihood_dist_;

        double final_rel_llikelihood_;
        double final_llikelihood_;
        double rel_llikelihood_dist_tol_;
        double llikelihood_dist_tol_;
        bool rel_llikelihood_flag_;
        bool llikelihood_flag_;

        bool convergence_test_likelihood(size_t, parameters, double, double);
        bool convergence_test(size_t, parameters);
        double rel_eucl_dist_sqr(parameters prev_params, parameters current_params);
        double eucl_dist_sqr(parameters prev_params, parameters current_params);
        bool distance_convergence(double distance_sqr,double rel_distance_sqr);
     //   chrono::high_resolution_clock::time_point start_timing();
     //  void end_timing(chrono::high_resolution_clock::time_point t1);
        double gaussian_dist(double R_n,int k);
        //void create_prob_vector(std::vector <double> &prob_vector1,double R_n);
        double generate_lambda_an_bn(double &a_n,double &b_n,double R_n);
        double generate_lambda_an_bn_likelihood(double &a_n,double &b_n,double R_n,double &local_likelihood);
        double generate_lambda_an_bn_2(double &a_n,double &b_n,double R_n);
        double generate_lambda_an_bn_2_vec(double &a_n,double &b_n,const double R_n);
        //double normal_pdf(double R_n,int k);
        parameters Expectation_Maximization_OneStep_OneData(double R_n,parameters starting);
        inline double normal_pdf(double R_n,int k);
        double eval_inferred_return(double timestep_fraction);
        inline double normal_pdf_no_const(double R_n,int k);
        inline double rand01();
        inline void print_params();
        inline void print_likelihood();
        inline void print_double(double x);
        inline void seed_random();
        inline void print_string(string x);
        void update_return_vec();
        size_t num_threads_;
        bool price_flag_;
        void vector_cleanup();
        bool likelihood_convergence(double llikelihood_dist,double rel_llikelihood_dist,size_t i);
};

#endif // EM_CLASS_H


