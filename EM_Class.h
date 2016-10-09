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
#include<chrono>
using namespace std;

const double PI = atan(1.0)*4;

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
        chrono::high_resolution_clock::time_point start_timing();
        void end_timing(chrono::high_resolution_clock::time_point t1);
        //    double thread_start_timing();
        //  void thread_end_timing(double t1);
        void load_R(std::string fileloc);
        std::vector<double> Expectation_Maximization();
        double gaussian_dist(double R_n,int k);
        //void create_prob_vector(std::vector <double> &prob_vector1,double R_n);
        double generate_lambda_an_bn(double &a_n,double &b_n,double R_n);
        double generate_lambda_an_bn_2(double &a_n,double &b_n,double R_n);
        //double normal_pdf(double R_n,int k);
        inline double normal_pdf(double R_n,int k);
        inline double normal_pdf_no_const(double R_n,int k);
        double incomp_log_likelihood();
        double incomp_log_likelihood_threaded_clean();
        vector<double> expected_num_of_jumps();
        void print_vec_to_file(vector<double> d);
        vector<double> auto_EM(bool random);
        inline double rand01();
        void set_max_iterations(size_t max_em_iterations);
        bool convergence_test(size_t, parameters);
    protected:
    private:
        std::vector <double> R;
        parameters initial;
        parameters maximising;
        size_t max_poisson_terms_;
        int debug_level;
        //size_t m;
        size_t max_window_size; 
        double final_log_likelihood;
        size_t iterations_completed;
        size_t max_em_iterations_;
};

#endif // EM_CLASS_H


