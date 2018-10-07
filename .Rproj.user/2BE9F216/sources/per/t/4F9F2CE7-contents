// [[Rcpp::depends(RcppArmadillo, RiemBase)]]

#include "RcppArmadillo.h"
#include "riemfactory.h"         // loading RiemBase's header-only library


// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//


// [[Rcpp::export]]
double loader(arma::mat X){
  return(norm_euclidean(X));
}