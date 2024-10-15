#ifndef __DEEP_HELMHOLTZ_GAN_H__
#define __DEEP_HELMHOLTZ_GAN_H__

// Deep Helmholtz Generative Adversarial Network 
// Using Helmholtz Decomposition and Variational Auto-Encoders 
// each latent variable is a spike train fitting a gaussian distribution
// the generative model is a spike train generator 
// the discriminative model is a softmax classifier
#include "info_helper.h"
#include "matrix.h"
#include "autoencoder.h"


using namespace std;
using namespace provallo;

template <typename T>
class deep_helmholtz_network : public variational_auto_encoder<T>
{

protected:
helmholtz_machine<T> generator;
softmax_classifier<T> discriminator;

std::vector<variational_auto_encoder<T>> generators_ensemble;
std::vector<softmax_classifier<T>> discriminators_ensemble;


public:

    deep_helmholtz_network(const std::matrix<real_t> &input,const std::matrix<real_t> &target,const std::vector<real_t> &labels) : variational_auto_encoder<T>(input,target,labels)  {

        this->initialize();
    }
    void initialize()
    {

        this->generator = helmholtz_machine<T>();
        this->discriminator = softmax_classifier<T>();  

    }

    void init()
    {

    
    }

    

};

#endif 