#ifndef PARAMETERS_H
#define PARAMETERS_H

#include<vector>

class parameters
{
    friend class EM_Class;
    public:
    private:
    parameters();
    double sigma_s;
    double mu;
    double nu;
    double lambda;
    double tau_s;
    std::vector<double> get_params(){
        std::vector<double> param(5);
        param.at(0) = mu;
        param.at(1) = nu;
        param.at(2) = sigma_s;
        param.at(3) =tau_s;
        param.at(4) =lambda;

        return param;

    }

    protected:
    // private:
};

#endif // PARAMETERS_H
