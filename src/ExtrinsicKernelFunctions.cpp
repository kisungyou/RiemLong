#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;


arma::uvec SubVec(arma::uvec myVec, int start, int end){
  //Subset an integer vector in a continuous range
  //
  //Args:
  // myVec: arma::uvec, length p. Vector to be subset.
  // start: int. Must be positive. Starting index.
  // end: int. Must be less than p, end of index.
  //
  //Returns:
  // retVec: arma::uvec, length start-end +1. Subsetted vector
  /////////////////////////////////////////////////////////////////////////////
  
  int myLength = end - start + 1;
  
  arma::uvec retVec(myLength);
  
  for(int i = 0; i < myLength; i++){
    
    retVec(i) = myVec(start + i);
    
  }
  
  return(retVec);
  
}

// [[Rcpp::depends(RcppArmadillo)]]
arma::vec SubVec(arma::vec myVec, int start, int end){
  //Subset a vector in a continuous range
  //
  //Args:
  // myVec: arma::vec, length p. Vector to be subset.
  // start: int. Must be positive. Starting index.
  // end: int. Must be less than p, end of index.
  //
  //Returns:
  // retVec: arma::vec, length start-end +1. Subsetted vector
  /////////////////////////////////////////////////////////////////////////////
  
  int myLength = end - start + 1;
  
  arma::vec retVec(myLength);
  
  for(int i = 0; i < myLength; i++){
    
    retVec(i) = myVec(start + i);
    
  }
  
  return(retVec);
  
}


// [[Rcpp::depends(RcppArmadillo)]]
arma::uvec AntiSubVec(arma::uvec myVec, int start, int end){
  //Subset an integer vector outside of the continuous range provided
  //
  //Args:
  // myVec: arma::uvec, length p. Vector to be subset.
  // start: int. Must be positive. Starting index.
  // end: int. Must be less than p, end of index.
  //
  //Returns:
  // retVec: arma::uvec, length start-end +1. Subsetted vector
  /////////////////////////////////////////////////////////////////////////////
  
  int p = myVec.n_elem;
  
  int myLength = p - (end - start + 1);
  
  arma::uvec retVec(myLength);
  
  //If start is greater than 0 and end is less than the length
  //pull values from beginning then the end
  
  if( (start > 0) && (end < (p-1))){
    
   for(int i = 0; i < start; i++){
     
     retVec(i) = myVec(i);
     
   }
   
   for(int i = (end + 1); i < p; i++){
   
     retVec(i - (end-start + 1)) = myVec(i);
     
   }
    
  }else{
    
    if( (start == 0) && (end < (p - 1)) ){
      
      retVec = SubVec(myVec, end + 1, p-1);
      
    }
    
    if( (start > 0) && (end == (p-1)) ){
      
      retVec = SubVec(myVec, 0, start  - 1);
      
    }
    
    if( (start == 0) && (end == (p - 1)) ){
      
      retVec = SubVec(myVec, 0, p - 1);
      
    }
     
  }
  
  return(retVec);
  
}

// [[Rcpp::depends(RcppArmadillo)]]
arma::vec AntiSubVec(arma::vec myVec, int start, int end){
  //Subset a vector outside of the continuous range provided
  //
  //Args:
  // myVec: arma::vec, length p. Vector to be subset.
  // start: int. Must be positive. Starting index.
  // end: int. Must be less than p, end of index.
  //
  //Returns:
  // retVec: arma::vec, length start-end +1. Subsetted vector
  /////////////////////////////////////////////////////////////////////////////
  
    
  int p = myVec.n_elem;
  
  int myLength = p - (end - start + 1);
  
  arma::vec retVec(myLength);
  
  //If start is greater than 0 and end is less than the length
  //pull values from beginning then the end
  
  if( (start > 0) && (end < (p-1))){
    
   for(int i = 0; i < start; i++){
     
     retVec(i) = myVec(i);
     
   }
   
   for(int i = (end + 1); i < p; i++){
   
     retVec(i - (end-start + 1)) = myVec(i);
     
   }
    
  }else{
    
    if( (start == 0) && (end < (p - 1)) ){
      
      retVec = SubVec(myVec, end + 1, p-1);
      
    }
    
    if( (start > 0) && (end == (p-1)) ){
      
      retVec = SubVec(myVec, 0, start  - 1);
      
    }
    
    if( (start == 0) && (end == (p - 1)) ){
      
      retVec = SubVec(myVec, 0, p - 1);
      
    }
     
  }
  
  return(retVec);
  
  
}





// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List timesTwo(int x) {
  return(Rcpp::List::create(Rcpp::Named("result") = x * 2,
                            Rcpp::Named("input") = x));
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double GaussKern(arma::vec x, arma::vec y, double h) {
  // Returns the gaussian kernel of vectors x and y with the given badnwidth h
  //
  //INPUT:
  // x: arma::vec x. First vector to perform kernel distance with.
  // y: arma::vec y. Second vector to perform kernel distance with.
  // h: double. Bandwidth for kernel evaluation.
  //
  //OUTPUT:
  // ret: double. The value of the gaussian distance kernel between x and y
  //      with bandiwdth h.
  /////////////////////////////////////////////////////////////////////////////
  //initialize return value
  double ret;

  ret = exp(-sum(pow(x/h - y/h, 2)))/h;
  
  return(ret);
  
  
}



// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec FEuc(arma::vec x, arma::mat X, arma::mat Y, double h = 1){
  //Gives the Nadarya-Watson kernel estimate of x, given the data X and Y.
  //This computes the estimator as stated in the EKF paper
  //
  //Args:
  //  x: arma Vector, length p. Location the estimate is desired.
  //  X: arma Matrix, p x n. n is the number of observations, p is 
  //     the number of features. This is for training the estimate
  //  Y: arma Matrix, d x n. d is the dimension of the manifold 
  //     response.
  //  h: Double. Bandwidth parameter for kernel function.
  //     DEFAULT is 1.0.
  //
  //Returns: arma vector, length d. The Nadarya-Watson Kernel
  //         estimate at x, along the manifold.
  /////////////////////////////////////////////////////////////////////////////
  
  //Initialize Kernel variables
  int n = X.n_cols;
  
  int d = Y.n_rows;
  
  arma::vec kern_est = arma::zeros<arma::vec>(d);
  
  double kern_sum = 0;
  

  //Calculate denominator for each term
  for(int i = 0; i < n; i++) {
    
    kern_sum = kern_sum + GaussKern( x, X.col(i), h);
    
  }
  
  for(int i = 0; i < n; i++) {
    
    kern_est = kern_est + Y.col(i) * GaussKern( x, X.col(i), h);
    
  }
  
  return(kern_est);

}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat FEuc2(arma::vec x, arma::mat X, arma::mat Y, double h = 1){
  //Gives the Nadarya-Watson estimate for a 2-dimensional or matrix valued
  //response variable (i.e. planar shape, positive definite matrices). The
  //kernel is the Gaussian kernel.
  //
  //Args:
  //  x: arma::vec, length p. Feature vector where the estimate is to
  //     be made.
  //  X: arma::mat, p x n. Training matrix giving the features associated
  //     with each observation (covariate matrix).
  //  Y: arma::mat, d x (q * n). Matrix value response data on the manifold
  //     d x q is the dimension of the matrices (frequently d x d). Since cube
  //     is not supported by all distributions of Rcpp, the structure is
  //     the slices of the cube are binded columnwise. It's required that
  //     the row dimension d >= q, (rows must be greater or equal to column).
  //     Note to create objects like this, cbind can be used on the response
  //     variables or transpose of the response variables (as opposed to 
  //     rbind which would return the incorrect estimate object for this
  //     function).
  //  h: double. The bandiwdth parameter to set a smoothness to the kernel.
  //     DEFAULT is 1.0.
  //
  //Returns:
  //  kern_est: arma::mat, d x q. The Nadarya-Watson estimate made at x.
  /////////////////////////////////////////////////////////////////////////////
  
  //Retrieve relevant sizes of objects
  
  int n = X.n_cols;
  
  int d = Y.n_rows;
  
  int q = Y.n_cols/n;
  
  //Compute denominator for each term
  
  double denom_sum = 0;
  
  for(int i = 0; i < n; i++){
    
    R_CheckUserInterrupt();
    
    denom_sum = denom_sum + GaussKern( x, X.col(i), h);
    
  }
  
  //Compute kernel estimate
  arma::mat kern_est = zeros(d, q);  
  
  for(int i = 0; i < n; i++){

    R_CheckUserInterrupt();
    
    //Rcpp::Rcout << "i is "<< i <<std::endl;
    kern_est = kern_est + Y(span::all, span(i*q, (i+1)*q-1))
    * GaussKern(x, X.col(i), h) / denom_sum;
        
  }

  return(kern_est);
  
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec FSphere(arma::vec x, arma::mat X, arma::mat Y, arma::vec center, 
                  double h = 1, double radius = 1){
  //Returns the extrinsic kernel regression estimate for sphere-valued
  //response variables at a point x
  //
  //Args:
  // x: arma::vec, length p. The value at which the estimate is to be
  //    made.
  // X: arma::mat, p x n. n is the number of observations. The covariate
  //    matrix with which the estimate is to be trained.
  // Y: arma::mat, d x n. The sphere-valued response vectors.
  // h: double. The bandwidth for the kernel function. DEFAULT is 1.
  // radius: double. The radius of the sphere on which the response lies.
  // center: arma::vec, length d. The center of the sphere that the response
  //         lie on. No default.
  //
  //Returns: An estimate for the response at x on the sphere with the
  //         specified radius and center, calculated by the inclusion 
  //         map on the sphere
  /////////////////////////////////////////////////////////////////////////////

  arma::vec est = FEuc(x, X, Y, h);
  
  arma::vec denomvec = sqrt(sum(pow(est,2)));
  
  double denom = denomvec(0);
  
  return(center + radius * (est/denom));
  
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat SpherePredict(arma::mat test, arma::mat train, arma::mat Y,
                        arma::vec center, double radius = 1,
                        double h = 1){
  //Predicts sphere-responses for multiple observations using the extrinsic
  //kernel regression method. The map is the inclusion map for the sphere.
  //
  //Args:
  // test: arma::mat, p x m. Multiple observations to be predicted
  // train: arma::mat, p x n. Observations for training
  // Y: arma::mat, d x n. Sphere response of training observations
  // radius: double. Radius of the sphere that the response is on.
  // center: arma::vec, length d. Center of the sphere the response is on.
  // h: double. Bandwidth for the kernel regression function. DEFAULT is 1.
  //
  //Returns:
  //  arma::mat, d x m. EKR predictions for the test observations.
  /////////////////////////////////////////////////////////////////////////////

  int m = test.n_cols;
  
  int d = Y.n_rows;
  
  arma::mat predict = zeros(d,m);
  
  for(int i = 0; i < m; i++){
    
    predict.col(i) = FSphere(test.col(i), train, Y, center, h, radius);
    
  }
  
  return(predict);
  
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double SphereDist(arma::vec y1, arma::vec y2, arma::vec center, double radius){
  //Geodesic distance between the two points on the specified sphere
  //
  //Args:
  // y1: arma::vec, length m. The first point on the sphere
  // y2: arma::vec length m. The second point on the sphere
  // center: arma::vec length m. The center of the sphere on which
  //         the points lie.
  // radius: double. The radius of the sphere.
  //
  //Returns: Double. Geodesic distance between the two points provided.
  /////////////////////////////////////////////////////////////////////////////
  
  //Center the vectors
  arma::vec y1c = y1 - center;
  
  arma::vec y2c = y2 - center;                 
  
  double angle = acos(accu(y1c%y2c) /
                 (sqrt(accu(pow(y1c,2))) * 
                 sqrt(accu(pow(y2c,2)))));
  
  double dist = radius * angle;
  
  return(dist);
       
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double GetSphereMSE(arma::mat pred, arma::mat val, arma::vec center,
                    double radius){
  //Gets the MSE of a set of predictions from the true sphere values
  //on a sphere with the given radius and center. Used by the 
  //SphereCrossValPredict function
  //
  //Args:
  // pred: arma::mat d x m. Predicted values.
  // val: arma::mat d x m. True/validation values.
  // center: arma::vec d. Center of sphere
  // radius: double. Radius of sphere.
  //
  //Returns:
  // mse: double. MSE of the values
  /////////////////////////////////////////////////////////////////////////////
  
  int m = pred.n_cols;
  
  double mse = 0;
  
  for(int i = 0; i < m; i++){
    
    mse += pow(SphereDist(pred.col(i), val.col(i), center, radius), 2);
    
  }
  
  return(mse / m);
                      
}



// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List SphereCrossValPredict(arma::mat x, arma::mat X, arma::mat Y,
                                 arma::vec center, double radius,
                                 double h_low = .01, double h_high = 2,
                                 double h_by = .1, int k = 10, 
                                 bool talk = false){
  //Performs cross validation selection of the bandwidth for extrinsic kernel
  //regression, and then predicts the sphere response for a given test set of
  //features. Uses the inclusion map for the sphere, geodesic distance for
  //residuals.
  //
  //Args:
  // x: arma::mat p x m. Test matrix of features for the prediction
  // X: arma::mat p x n. Training data set of features.
  // Y: arma::mat d x n. Training set of sphere-valued responses
  //    corresponding to the training set.
  // h_low: double. The lower bound for h to be tried for bandwidth selection.
  //        DEFAULT is .01.
  // h_high: double. The upper bound for h to be tried for bandwidth selection.
  //         DEFAULT is 2.
  // h_by: double. The intervals that the bandwidth should be tried at over
  //       the range. DEFAULT is .1.
  // k:    int. The number of cross validation sets to use.
  //
  //Returns:
  // prediction: arma::mat  d x m. The predicted sphere response for the test
  //             observations contained in x.
  // h: double. The selected bandwidth from cross validation used to make the
  //    prediction.
  // MSE: arma::mat. The MSE from all the cross validation trials.
  // h_seq: arma::vec. The grid of values used in the cross validation 
  //        search.
  /////////////////////////////////////////////////////////////////////////////
  
  int n = X.n_cols;
  
  //int p = X.n_rows;
  
  //Get the number of observations to go into each matrix
  
  int n_cv = (int)n / k;
  
  //int rem_cv = n % k;
  
  //Initialize the indexing list
  
  arma::uvec obs_ind((n_cv*k));
  
  for(int i = 0; i < (n_cv * k); i++){
    
    obs_ind(i) = i;
    
  }
  
  obs_ind = shuffle(obs_ind);
  
  //Initialize the bandwidth sequence
  
  int hsteps = (h_high - h_low) / h_by + 2;
  
  arma::vec h_seq(hsteps);
  
  for(int ii = 0; ii < hsteps; ii++){
    
    h_seq(ii) = h_low + ii*h_by;
    
  }

  //initialize the MSE matrix
  
  arma::mat mse_mat(k, hsteps);
  
  
  //For each of the bandwidth values Find the MSE using each of the 
  //k subsamples as the validation set
  arma::mat cv_Xval;
     
  arma::mat cv_Xtrain;
     
  arma::mat cv_Yval;
     
  arma::mat cv_Ytrain;
  //Pick a validation
  for(int kk = 0; kk < k; kk++){
    
    if(talk){
      Rcpp::Rcout << "Checking Validation Group " << kk + 1 << std::endl;
    }
    
    //Get the subset
    if(kk == (k-1)){
      
     uvec cv_val_ind = SubVec(obs_ind, n_cv * (kk), n - 1);
     
     uvec cv_train_ind = AntiSubVec(obs_ind, n_cv * (kk), n - 1);
     
     cv_Xval = X.cols(cv_val_ind);

     cv_Xtrain = X.cols(cv_train_ind);
     
     cv_Yval = Y.cols(cv_val_ind);
     
     cv_Ytrain = Y.cols(cv_train_ind);
     
    } else {
     
     uvec cv_val_ind = SubVec(obs_ind, n_cv * kk, n_cv * (kk + 1) - 1);
     
     uvec cv_train_ind = AntiSubVec(obs_ind, n_cv * kk, n_cv * (kk + 1) - 1);
     
     cv_Xval = X.cols(cv_val_ind);
     
     cv_Xtrain = X.cols(cv_train_ind);
     
     cv_Yval = Y.cols(cv_val_ind);
     
     cv_Ytrain = Y.cols(cv_train_ind);
     
    }
    
    //For the validation set, check each of the bandwidth values
    
    for(int hh = 0; hh < hsteps; hh++){
      
      R_CheckUserInterrupt();
      
      //get the predicted values, will be d x (training set size)
      
      arma::mat val_predict = SpherePredict(cv_Xval, cv_Xtrain, cv_Ytrain,
                                            center, radius, h_seq(hh));
      
      mse_mat(kk, hh) = GetSphereMSE(val_predict, cv_Yval, center, radius);
      
      
    }
    
  }
  
  //Select a bandwidth value based on the performance of each
  
  arma::rowvec total_mse = sum(mse_mat);
  
  arma::uvec band_ind = find(total_mse == min(total_mse));
  
  double h = h_seq(band_ind(0));
  
  arma::mat prediction = SpherePredict(x, X, Y, center, radius, h);
  
  return(Rcpp::List::create(Rcpp::Named("prediction") = prediction,
                            Rcpp::Named("h") = h,
                            Rcpp::Named("MSE") = mse_mat,
                            Rcpp::Named("hseq") = h_seq));
  
  
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat SphereCrossValPredictQuiet(arma::mat x, arma::mat X, arma::mat Y,
                                 arma::vec center, double radius,
                                 double h_low = .01, double h_high = 2,
                                 double h_by = .1, int k = 10){
  //Performs cross validation selection of the bandwidth for extrinsic kernel
  //regression, and then predicts the sphere response for a given test set of
  //features. Uses the inclusion map for the sphere, geodesic distance for
  //residuals. Returns only the prediction for the test sets.
  //
  //Args:
  // x: arma::mat p x m. Test matrix of features for the prediction
  // X: arma::mat p x n. Training data set of features.
  // Y: arma::mat d x n. Training set of sphere-valued responses
  //    corresponding to the training set.
  // h_low: double. The lower bound for h to be tried for bandwidth selection.
  //        DEFAULT is .01.
  // h_high: double. The upper bound for h to be tried for bandwidth selection.
  //         DEFAULT is 2.
  // h_by: double. The intervals that the bandwidth should be tried at over
  //       the range. DEFAULT is .1.
  // k:    int. The number of cross validation sets to use.
  //
  //Returns:
  // prediction: arma::mat  d x m. The predicted sphere response for the test
  //             observations contained in x.
  /////////////////////////////////////////////////////////////////////////////
  
  int n = X.n_cols;
  
  //int p = X.n_rows;
  
  //Get the number of observations to go into each matrix
  
  int n_cv = (int)n / k;
  
  //int rem_cv = n % k;
  
  //Initialize the indexing list
  
  arma::uvec obs_ind((n_cv*k));
  
  for(int i = 0; i < (n_cv * k); i++){
    
    obs_ind(i) = i;
    
  }
  
  obs_ind = shuffle(obs_ind);
  
  //Initialize the bandwidth sequence
  
  int hsteps = (h_high - h_low) / h_by + 2;
  
  arma::vec h_seq(hsteps);
  
  for(int ii = 0; ii < hsteps; ii++){
    
    h_seq(ii) = h_low + ii*h_by;
    
  }

  //initialize the MSE matrix
  
  arma::mat mse_mat(k, hsteps);
  
  
  //For each of the bandwidth values Find the MSE using each of the 
  //k subsamples as the validation set
  arma::mat cv_Xval;
     
  arma::mat cv_Xtrain;
     
  arma::mat cv_Yval;
     
  arma::mat cv_Ytrain;
  //Pick a validation
  for(int kk = 0; kk < k; kk++){
    
    //Get the subset
    if(kk == (k-1)){
      
     uvec cv_val_ind = SubVec(obs_ind, n_cv * (kk), n - 1);
     
     uvec cv_train_ind = AntiSubVec(obs_ind, n_cv * (kk), n - 1);
     
     cv_Xval = X.cols(cv_val_ind);

     cv_Xtrain = X.cols(cv_train_ind);
     
     cv_Yval = Y.cols(cv_val_ind);
     
     cv_Ytrain = Y.cols(cv_train_ind);
     
    } else {
     
     uvec cv_val_ind = SubVec(obs_ind, n_cv * kk, n_cv * (kk + 1) - 1);
     
     uvec cv_train_ind = AntiSubVec(obs_ind, n_cv * kk, n_cv * (kk + 1) - 1);
     
     cv_Xval = X.cols(cv_val_ind);
     
     cv_Xtrain = X.cols(cv_train_ind);
     
     cv_Yval = Y.cols(cv_val_ind);
     
     cv_Ytrain = Y.cols(cv_train_ind);
     
    }
    
    //For the validation set, check each of the bandwidth values
    
    for(int hh = 0; hh < hsteps; hh++){
      
      R_CheckUserInterrupt();
      
      //get the predicted values, will be d x (training set size)
      
      arma::mat val_predict = SpherePredict(cv_Xval, cv_Xtrain, cv_Ytrain,
                                            center, radius, h_seq(hh));
      
      mse_mat(kk, hh) = GetSphereMSE(val_predict, cv_Yval, center, radius);
      
      
    }
    
  }
  
  //Select a bandwidth value based on the performance of each
  
  arma::rowvec total_mse = sum(mse_mat);
  
  arma::uvec band_ind = find(total_mse == min(total_mse));
  
  double h = h_seq(band_ind(0));
  
  arma::mat prediction = SpherePredict(x, X, Y, center, radius, h);
  
  return( prediction );
  
  
}

// [[Rcpp::depends(RcppArmadillo)]]
arma::vec GetKernelWeights(arma::vec x, arma::mat X, double h){
  //Get the vector of kernel weights to be used by the gradient descent 
  //algorithm
  //
  //Args:
  // x: arma::vec, length p. The value to evaluate the kernel.
  // X: arma::mat, p x n. The functional values for the kernel.
  // h: double. The bandwidth to evaluate the kernel.
  //
  //Returns:
  // ret: arma::vec, length n. The kernel weight due to each
  //      value in X.
  /////////////////////////////////////////////////////////////////////////////
  
  int n = X.n_cols;
  
  arma::vec ret(n);
  
  for(int i = 0; i < n; i++){
    
    ret(i) = GaussKern(x, X.col(i), h);
    
  }
  
  ret = ret/accu(ret);
  
  return(ret);
}


// [[Rcpp::depends(RcppArmadillo)]]
arma::vec GradientStep(arma::vec y, arma::mat Y, arma::vec w, double s, int n){
  //Calculates a step of the gradient for the sphere intrinsic kernel 
  //regression model. Used by IntrinsicSphere function.
  //
  //Args:
  // y: arma::vec, length d. The current gradient location.
  // Y: arma::vec, d x n. The other training observations.
  // w: arma::vec, length n. The kernel weights associatied with each 
  //    observation.
  // s: double. The step size of the gradient algorithm
  // n: int. The number of training observations.
  //
  //Returns:
  // x: arma::vec, length d. New estimate.
  /////////////////////////////////////////////////////////////////////////////
  
  int d = y.n_elem;
  
  y = y/sqrt(accu(pow(y,2)));
  
  arma::vec x = zeros(d);
  
  for(int i = 0; i < n; i++){
    
    double yY = dot(y, Y.col(i));
  
    x = x + 2 * w(i) * acos(yY)/(sqrt(1-pow(yY,2))) * (Y.col(i) - yY * y);
    
  }
  
  //Since we are minimizing we climb down the gradient
  
   x = y + s * x;
   
   return(x);

}






// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec IntrinsicSphere(arma::vec x, arma::mat X, arma::mat Y,
                          arma::vec center, double radius, double eps = .001,
                          double h = .6,
                          double step = .01, int max_iter = 100000){
  //A gradient descent estimate of the intrinsic model for sphere regression.
  //The weights are calculated using the Gaussian kernel.
  //
  //Args:
  // x: arma::vec length p. The covariates for where the sphere estimate should
  //     be made.
  // X: arma::mat, p x n. A sample of n training examples.
  // Y: arma::mat, d x n. A sample of sphere (S^{d-1}) valued responses
  //    corresponding to the training examples in X.
  // center: arma::vec, length d. Center of the sphere on which the responses
  //         lay. 
  // radius: double. Radius of the sphere for the responses.
  // eps: double. The threshold for the convergence of the gradient descent.
  //      DEFAULT is .001.
  // h: double. The bandwidth for estimating the Gaussian kernel. Default is
  //    .6.
  // step: double. The step size for the gradient descent method.
  // max_iter: int. The maximum number of iterations the descent algorithm is
  //           allowed to take.
  /////////////////////////////////////////////////////////////////////////////
  
  int d = Y.n_rows;
  
  int n = Y.n_cols;
  
  arma::vec y_old = ones(d);
  
  arma::vec y_new = (y_old/sqrt(accu(pow(y_old,2)))) * sqrt(radius) + center;

  int i = 0;
  
  arma::vec kernel_weights = GetKernelWeights(x, X, h);
  
  while( (accu(abs(y_old - y_new)) > eps) && (i < max_iter)){
    
    
    y_old = y_new;
    
    y_new = GradientStep(y_old,Y, kernel_weights, step, n);
    
    i++;
    
  }
  
  return( y_new );
  
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat IntrinsicPredict(arma::mat test, arma::mat train, arma::mat Y,
                          arma::vec center, double radius, double eps = .001,
                          double h = .6,
                          double step = .01, int max_iter = 100000){
  //Predicts sphere-responses for multiple observations using the extrinsic
  //kernel regression method. The map is the inclusion map for the sphere.
  //
  //Args:
  // test: arma::mat, p x m. Multiple observations to be predicted
  // train: arma::mat, p x n. Observations for training
  // Y: arma::mat, d x n. Sphere response of training observations
  // radius: double. Radius of the sphere that the response is on.
  // center: arma::vec, length d. Center of the sphere the response is on.
  // eps: Double, Threshold for convergence. DEFAULT .001.
  // step: double. Step size for descent. DEFAULT .01.
  // max_iter: int. Maximum iterations for descent.
  // h: double. Bandwidth for the kernel regression function. DEFAULT is .6.
  //
  //Returns:
  //  arma::mat, d x m. Intrinsic predictions for the test observations.
  /////////////////////////////////////////////////////////////////////////////

  int m = test.n_cols;
  
  int d = Y.n_rows;
  
  arma::mat predict = zeros(d,m);
  
  for(int i = 0; i < m; i++){
    
    R_CheckUserInterrupt();
    
    predict.col(i) = IntrinsicSphere(test.col(i), train, Y, center, radius,
                                     eps, h, step, max_iter);
    
  }
  
  return(predict);
  
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List FSubspace(arma::vec x, arma::mat X, arma::mat Y, double h = 1){
  //Estimates a subspace response to dependent variables in x according
  //to a kernel regression using an extrinsic distance metric on the 
  //subspaces. 
  //
  //Args:
  // x: arma::vec, length p. The covariates for which the prediction is to be
  //    made.
  // X: arma::mat p x n. The set of covariates associated with n training 
  //    observations
  // Y: arma::mat d x (n * d). The set of projection matrix responses
  //    associated with the n training observations. Projection matrices
  //    are bound columnwise, and are the projection matrices onto the 
  //    the subspaces for which we are obtaining an estimate.
  // h: double. The bandwidth for the kernel.
  //
  //Returns: Rcpp::List
  // proj: arma::mat d x d. The projection matrix onto the subspace that is
  //       the extrinsic kernel estimate
  // sub: arma::mat d x k. k is the estimated dimension of the subspace
  //      and the columns are a basis for the estimated subspace.
  /////////////////////////////////////////////////////////////////////////////
  
  int d = Y.n_rows;
  
  //Obtain projection estimate
  arma::mat proj = FEuc2(x, X, Y, h);
  
  
  // Get the estimated rank of the estimate


  int rough_rank = round(accu(proj.diag()));
  
  arma::vec eigval;
  
  arma::mat eigvec;
  
  arma::eig_sym(eigval, eigvec, proj);
  
  //Get a basis for the estimated   
  
  arma::mat projsub = eigvec.cols(d - rough_rank , d - 1);
  
  proj = projsub * projsub.t();
  
  return(Rcpp::List::create(Rcpp::Named("proj") = proj,
                            Rcpp::Named("sub") = projsub));
  
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double ConwaySphereDistance(arma::mat p1, arma::mat p2, int d){
  //Calculates the Conway Sphere distance between two projection
  //matrices given the ambient dimension d
  //
  //Args:
  // p1: arma::mat, d x d. The first projection matrix.
  // p2: arma::mat, d x d. The second projection matrix.
  // d: int. The ambient dimension
  //
  //Rets:
  // ret: double. The distance between the two projection matrices.
  /////////////////////////////////////////////////////////////////////////////
  
  arma::vec v1(d*(d+1)/2);
  
  arma::vec v2(d*(d+1)/2);
  
  arma::vec eyevec(d*(d+1)/2);
  
  int vecind = 0;
  
  for(int c = 0; c < d; c++){
  
    for(int r = c; r < d; r++){
  
      v1(vecind) = p1(r,c);
    
        v2(vecind) = p2(r,c);
        
        if(r == c){
        
          eyevec(vecind) = 1.0/2.0;
      
        }else{
      
          eyevec(vecind) = 0;
      
        }
      
        vecind++;
     }
  
   }
  
   arma::vec v1p = v1 - eyevec;
   
   arma::vec v2p = v2 - eyevec;
   
   double ret = acos(dot(v1p,v2p)/(norm(v1p)*norm(v2p))) * pow(d,.5)/(float)2;
  
   return(ret);
  
  
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double ConwayLowerSphereDistance(arma::mat p1, arma::mat p2, int d, int k){
  //Calculates the Conway Sphere distance between two projection
  //matrices given the ambient dimension d on the lower dimensional
  //embedding (that is a function of subspace distance). This is
  //only applicable when the subspaces are the same dimension
  //
  //Args:
  // p1: arma::mat, d x d. The first projection matrix.
  // p2: arma::mat, d x d. The second projection matrix.
  // d: int. The ambient dimension
  // k: int. The dimension of both subspaces
  //
  //Rets:
  // ret: double. The distance between the two projection matrices.
  /////////////////////////////////////////////////////////////////////////////
  
  arma::vec v1(d*(d+1)/2 - 1);
  
  arma::vec v2(d*(d+1)/2 - 1);
  
  arma::vec eyevec(d*(d+1)/2 - 1);
  
  int vecind = 0;
  
  for(int c = 0; c < d; c++){
  
    for(int r = c; r < d; r++){
  
      if(vecind < (d*(d+1)/2 - 1)){
        //Rcpp::Rcout << "r is "<< r << " and c is "<< c<< " and vecind is "<<vecind <<std::endl;
        v1(vecind) = p1(r,c);
    
        v2(vecind) = p2(r,c);
        
        if(r == c){
        
          eyevec(vecind) = k/(float)d;
      
        }else{
      
          eyevec(vecind) = 0;
      
        }
      
        vecind++;
        
      }
      
     }
  
   }
  
   arma::vec v1p = v1 - eyevec;
   
   arma::vec v2p = v2 - eyevec;
   
   double ret = acos(dot(v1p,v2p)/(norm(v1p)*norm(v2p))) * pow(k*(d - k) / (float)d,.5);
  
   return(ret);
  
  
}



// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat SubspacePairDists(arma::mat x, int n, int d){
  //Calculates pairwise distances between n projection matrices according
  //to the greater circle distance on the conway sphere
  //Args:
  // x: arma::mat, d x (n * d). A matrix of projection matrices (m x m) of 
  //    subspaces
  // n: int. the number of projection matrices
  //
  //Returns:
  // ret: arma::mat, n x n. Matrix of pairwise conway sphere geodesic distances
  /////////////////////////////////////////////////////////////////////////////
  
  arma::mat ret(n,n);
  
  for(int i = 0; i < n; i++){
    
    for(int j = i; j < n; j++){
      
      if(i == j){
        
        ret(i,j) = 0;
        
      }else{
        
        ret(i,j) = ConwaySphereDistance(x.cols(i*d, ((i+1)*d-1)), 
                                        x.cols(j*d, ((j+1)*d-1)),
                                        d);
                                        
        ret(j,i) = ret(i,j); 
        
      }
    
    }
    
  }
  
  return(ret);
  
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List IntrinsicSphereCrossValPredict(arma::mat x, arma::mat X, arma::mat Y,
                                 arma::vec center, double radius,
                                 double h_low = .01, double h_high = 2,
                                 double h_by = .1, int k = 10, 
                                 bool talk = false){
  //Performs cross validation selection of the bandwidth for extrinsic kernel
  //regression, and then predicts the sphere response for a given test set of
  //features. Uses the inclusion map for the sphere, geodesic distance for
  //residuals.
  //
  //Args:
  // x: arma::mat p x m. Test matrix of features for the prediction
  // X: arma::mat p x n. Training data set of features.
  // Y: arma::mat d x n. Training set of sphere-valued responses
  //    corresponding to the training set.
  // h_low: double. The lower bound for h to be tried for bandwidth selection.
  //        DEFAULT is .01.
  // h_high: double. The upper bound for h to be tried for bandwidth selection.
  //         DEFAULT is 2.
  // h_by: double. The intervals that the bandwidth should be tried at over
  //       the range. DEFAULT is .1.
  // k:    int. The number of cross validation sets to use.
  //
  //Returns:
  // prediction: arma::mat  d x m. The predicted sphere response for the test
  //             observations contained in x.
  // h: double. The selected bandwidth from cross validation used to make the
  //    prediction.
  // MSE: arma::mat. The MSE from all the cross validation trials.
  // h_seq: arma::vec. The grid of values used in the cross validation 
  //        search.
  /////////////////////////////////////////////////////////////////////////////
  
  int n = X.n_cols;
  
  //int p = X.n_rows;
  
  //Get the number of observations to go into each matrix
  
  int n_cv = (int)n / k;
  
  //int rem_cv = n % k;
  
  //Initialize the indexing list
  
  arma::uvec obs_ind((n_cv*k));
  
  for(int i = 0; i < (n_cv * k); i++){
    
    obs_ind(i) = i;
    
  }
  
  obs_ind = shuffle(obs_ind);
  
  //Initialize the bandwidth sequence
  
  int hsteps = (h_high - h_low) / h_by + 2;
  
  arma::vec h_seq(hsteps);
  
  for(int ii = 0; ii < hsteps; ii++){
    
    h_seq(ii) = h_low + ii*h_by;
    
  }

  //initialize the MSE matrix
  
  arma::mat mse_mat(k, hsteps);
  
  
  //For each of the bandwidth values Find the MSE using each of the 
  //k subsamples as the validation set
  arma::mat cv_Xval;
     
  arma::mat cv_Xtrain;
     
  arma::mat cv_Yval;
     
  arma::mat cv_Ytrain;
  //Pick a validation
  for(int kk = 0; kk < k; kk++){
    
    if(talk){
      Rcpp::Rcout << "Checking Validation Group " << kk + 1 << std::endl;
    }
    
    //Get the subset
    if(kk == (k-1)){
      
     uvec cv_val_ind = SubVec(obs_ind, n_cv * (kk), n - 1);
     
     uvec cv_train_ind = AntiSubVec(obs_ind, n_cv * (kk), n - 1);
     
     cv_Xval = X.cols(cv_val_ind);

     cv_Xtrain = X.cols(cv_train_ind);
     
     cv_Yval = Y.cols(cv_val_ind);
     
     cv_Ytrain = Y.cols(cv_train_ind);
     
    } else {
     
     uvec cv_val_ind = SubVec(obs_ind, n_cv * kk, n_cv * (kk + 1) - 1);
     
     uvec cv_train_ind = AntiSubVec(obs_ind, n_cv * kk, n_cv * (kk + 1) - 1);
     
     cv_Xval = X.cols(cv_val_ind);
     
     cv_Xtrain = X.cols(cv_train_ind);
     
     cv_Yval = Y.cols(cv_val_ind);
     
     cv_Ytrain = Y.cols(cv_train_ind);
     
    }
    
    //For the validation set, check each of the bandwidth values
    
    for(int hh = 0; hh < hsteps; hh++){
      
      R_CheckUserInterrupt();
      
      //get the predicted values, will be d x (training set size)
      
      arma::mat val_predict = IntrinsicPredict(cv_Xval, cv_Xtrain, cv_Ytrain,
                                            center, radius, h_seq(hh));
      
      mse_mat(kk, hh) = GetSphereMSE(val_predict, cv_Yval, center, radius);
      
      
    }
    
  }
  
  //Select a bandwidth value based on the performance of each
  
  arma::rowvec total_mse = sum(mse_mat);
  
  arma::uvec band_ind = find(total_mse == min(total_mse));
  
  double h = h_seq(band_ind(0));
  
  arma::mat prediction = IntrinsicPredict(x, X, Y, center, radius, h);
  
  return(Rcpp::List::create(Rcpp::Named("prediction") = prediction,
                            Rcpp::Named("h") = h,
                            Rcpp::Named("MSE") = mse_mat,
                            Rcpp::Named("hseq") = h_seq));
  
  
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::cx_vec KendallsExponentialMap_x(arma::cx_vec x, arma::cx_vec v){
  //Computes the exponential map in kendalls shape space.
  //The vectors are assumed to be elements of C^{k-1} such that 
  //translation is removed. The map to be computed here is
  //cos(theta)*x  + ||x||sin(theta)/theta * v, theta=||v||.
  //Args:
  //x: cx_vec, length k. The shape specifying the space of tangent vectors
  //   v to compute the exponential map for.
  //v: cx_vec, length k. The tangent vector to x to compute the vector for.
  //Returns
  //ret: cx_vec, length k. The exponential map.
  /////////////////////////////////////////////////////////////////////////////
  
  double theta_ = norm(v);
  
  int k = v.n_elem;
  
  vec theta__(k);
  
  theta__.fill(theta_);
    
  vec zer; zer.zeros(k);
  
  cx_vec theta(theta__,zer);
  
  cx_vec ret = cos(theta)% x + (norm(x)*sin(theta)/theta)%v;
  
  return(ret);
  
}

// [[Rcpp::export]]
arma::cx_vec KendallsLogMap_x(arma::cx_vec x,arma::cx_vec y){
  //Computes the log map in kendalls shape space.
  //The vectors are assumed to be elements of C^{k-1} such that 
  //translation is removed. The map to be computed here is
  //[theta*(y - proj_x(y))]/[||y-proj_x(y)||] 
  //theta=arccos[|(<x,y>)|/(||x|| ||y||)].
  //proj_x(y) = x*<x,y>/||x||^2
  //Args:
  //x: cx_vec, length k. The shape specifying the space of tangent vectors
  //   v to compute the log map for.
  //v: cx_vec, length k. The tangent vector to x to compute the vector for.
  //Returns
  //ret: cx_vec, length k. The log map.
  /////////////////////////////////////////////////////////////////////////////
  //Rcpp::Rcout<< "Here 1" << std::endl;  
  cx_vec inner_prod_xyvec = trans(y)*x;
  //Rcpp::Rcout<< "Here 2" << std::endl;
  cx_double inner_prod_val = inner_prod_xyvec(0);
  //Rcpp::Rcout<< "Here 3" << std::endl;
  double theta = acos(abs(inner_prod_val)/(norm(x)*norm(y)));
  
  //Rcpp::Rcout<< "Here 4" << std::endl;
  
  cx_vec proj_y_on_x = x*(inner_prod_val)/pow(norm(x),2);
  //Rcpp::Rcout<< "Here 5" << std::endl;
  cx_vec ret = (theta*(y - proj_y_on_x))/norm(y-proj_y_on_x);
  
  return(ret);
  
}




arma::cx_vec GradientStepShape(arma::cx_vec y, arma::cx_mat Y, arma::vec w, double s, int n){
  //Calculates a step of the gradient for the sphere intrinsic kernel 
  //regression model. Used by IntrinsicSphere function.
  //
  //Args:
  // y: arma::vec, length d. The current gradient location.
  // Y: arma::vec, d x n. The other training observations.
  // w: arma::vec, length n. The kernel weights associatied with each 
  //    observation.
  // s: double. The step size of the gradient algorithm
  // n: int. The number of training observations.
  //
  //Returns:
  // x: arma::vec, length d. New estimate.
  /////////////////////////////////////////////////////////////////////////////
  
  int d = y.n_elem;
  
  y = y/norm(y);
  //Are these pre shapes?
  
  cx_vec x = zeros<cx_vec>(d);
  
  for(int i = 0; i < n; i++){
    
    //cx_double yY = dot(y, Y.col(i));
    
    x = x + 2 * w(i) * KendallsLogMap_x(Y.col(i), y);
    
  }
  
  //Since we are minimizing we climb down the gradient
  
  x = y + s * x;
  
  return(x);
  
}







// [[Rcpp::export]]
arma::cx_vec IntrinsicShape(arma::vec x, arma::mat X, arma::cx_mat Y,
                          arma::cx_vec center, double radius, double eps = .001,
                          double h = .6,
                          double step = .01, int max_iter = 100000){
  //A gradient descent estimate of the intrinsic model for sphere regression.
  //The weights are calculated using the Gaussian kernel.
  //
  //Args:
  // x: arma::vec length p. The covariates for where the sphere estimate should
  //     be made.
  // X: arma::mat, p x n. A sample of n training examples.
  // Y: arma::mat, d x n. A sample of sphere (S^{d-1}) valued responses
  //    corresponding to the training examples in X.
  // center: arma::cx_vec, length d. Center of the sphere on which the responses
  //         lay. 
  // radius: double. Radius of the sphere for the responses.
  // eps: double. The threshold for the convergence of the gradient descent.
  //      DEFAULT is .001.
  // h: double. The bandwidth for estimating the Gaussian kernel. Default is
  //    .6.
  // step: double. The step size for the gradient descent method.
  // max_iter: int. The maximum number of iterations the descent algorithm is
  //           allowed to take.
  /////////////////////////////////////////////////////////////////////////////
  
  int d = Y.n_rows;
  
  int n = Y.n_cols;
  
  arma::cx_vec y_old = ones<cx_vec>(d);
  
  arma::cx_vec y_new = (y_old/norm(y_old)) * sqrt(radius) + center;
  
  int i = 0;
  
  arma::vec kernel_weights = GetKernelWeights(x, X, h);
  
  while( (accu(norm(y_old - y_new)) > eps) && (i < max_iter)){
    
    
    y_old = y_new;
    
    y_new = GradientStepShape(y_old,Y, kernel_weights, step, n);
    
    i++;
    
  }
  
  return( y_new );
  
}


// [[Rcpp::export]]
arma::cx_mat IntrinsicPredictShape(arma::mat test, arma::mat train, arma::cx_mat Y,
                           arma::cx_vec center, double radius, double eps = .001,
                           double h = .6,
                           double step = .01, int max_iter = 100000){
  //Predicts sphere-responses for multiple observations using the extrinsic
  //kernel regression method. The map is the inclusion map for the sphere.
  //
  //Args:
  // test: arma::mat, p x m. Multiple observations to be predicted
  // train: arma::mat, p x n. Observations for training
  // Y: arma::mat, d x n. Sphere response of training observations
  // radius: double. Radius of the sphere that the response is on.
  // center: arma::vec, length d. Center of the sphere the response is on.
  // eps: Double, Threshold for convergence. DEFAULT .001.
  // step: double. Step size for descent. DEFAULT .01.
  // max_iter: int. Maximum iterations for descent.
  // h: double. Bandwidth for the kernel regression function. DEFAULT is .6.
  //
  //Returns:
  //  arma::mat, d x m. Intrinsic predictions for the test observations.
  /////////////////////////////////////////////////////////////////////////////
  
  int m = test.n_cols;
  
  int d = Y.n_rows;
  
  arma::cx_mat predict = zeros<cx_mat>(d,m);
  
  for(int i = 0; i < m; i++){
    
    R_CheckUserInterrupt();
    
    predict.col(i) = IntrinsicShape(test.col(i), train, Y, center, radius,
                eps, h, step, max_iter);
    
  }
  
  return(predict);
  
}

arma::cx_mat ConvertRealToComplex(arma::mat x){
  //Converts a real matrix to a complex matrix.
  //Args:
  // x: mat. k x (n * 2);
  //Returns:
  //ret: cx_mat. k x n. 
  /////////////////////////////////////////////////////////////////////////////
  
  int k = x.n_rows;
  int n2 = x.n_cols;
  int n = n2/2;
  
  arma::mat the_real(k,n);
  arma::mat the_imaginary(k,n);
  
  for(int i = 0; i < n; i++){
    the_real.col(i) = x.col(i*2 + 1);
    the_imaginary.col(i) = x.col(i*2);
  }
  
  arma::cx_mat ret(the_real, the_imaginary);
  
  return(ret);
  
}


arma::mat ConvertComplexToReal( arma::cx_mat x){
  //Converts a complex matrix of the form k x n and makes it a 
  //matrix of the form k x (2 * n).
  //Args:
  // x: cx_mat, k x n.
  //Returns
  // ret:  mat. k x (2 * n);
  /////////////////////////////////////////////////////////////////////////////
  int k = x.n_rows;
  
  int n = x.n_cols;
  
  arma::mat ret(k, n * 2);
  
  arma::mat a = arma::imag(x);
  
  arma::mat b = arma::real(x);
  
  for(int i = 0; i < n; i++){
    
    ret.col(i*2) = a.col(i);
    
    ret.col(i*2 + 1) = b.col(i);
    
  }
  
  return(ret);
  
}

arma::mat ConvertComplexToReal(arma::cx_vec x){
  //Converts a complex vector of length k to a real matrix of dimension
  //k x 2.
  //Args:
  //x: cx_vec. length k.
  //Returns
  //ret: matrix. k x 2. Double.
  /////////////////////////////////////////////////////////////////////////////
  
  int k = x.n_elem;
  
  arma::mat ret(k, 2);
  
  ret.col(0) = arma::imag(x);
  
  ret.col(1) = arma::real(x);
  
  return(ret);
  
}

// [[Rcpp::export]]
arma::cx_vec FShape(arma::vec x, arma::mat X, arma::cx_mat Y, double h = 1){
  //Gets the extrinsic estimate for a shape, via the function
  //FEuc2. FEuc2 takes it's input in a particular way, such that
  //the complex values are split into real values. The format
  //will be k x (2 * n). This function primarily splits the complex values
  //and then feeds it into the FEuc2 solver.
  //Args:
  // x: arma::colvec, double, length p. The covariates to do a prediction
  // X: arma::mat, double, p x n. The covariates for the training observations;
  // Y: arma::cx_mat. k x n. The training observations.
  // h: double. The kernel bandwidth.
  //returns:
  // ret: arma::cx_vec, length k. Complex representation of the estimated value
  /////////////////////////////////////////////////////////////////////////////
  
  arma::mat Y_real = ConvertComplexToReal(Y);
  
  arma::mat estimate = FEuc2(x, X, Y_real, h);
  
  arma::cx_mat estimate_conv = ConvertRealToComplex(estimate);
  
  arma::cx_vec ret = estimate_conv.col(0);
  
  return(ret);
  
}

// [[Rcpp::export]]
arma::cx_mat FPredictShape(arma::mat test, arma::mat train, arma::cx_mat Y,
                     arma::cx_vec center, double radius, 
                     double h = .6){
  //Predicts sphere-responses for multiple observations using the extrinsic
  //kernel regression method. The map is the inclusion map for the sphere.
  //
  //Args:
  // test: arma::mat, p x m. Multiple observations to be predicted
  // train: arma::mat, p x n. Observations for training
  // Y: arma::mat, d x n. Sphere response of training observations
  // radius: double. Radius of the sphere that the response is on.
  // center: arma::vec, length d. Center of the sphere the response is on.
  // eps: Double, Threshold for convergence. DEFAULT .001.
  // step: double. Step size for descent. DEFAULT .01.
  // max_iter: int. Maximum iterations for descent.
  // h: double. Bandwidth for the kernel regression function. DEFAULT is .6.
  //
  //Returns:
  //  arma::mat, d x m. Intrinsic predictions for the test observations.
  /////////////////////////////////////////////////////////////////////////////
  
  int m = test.n_cols;
  
  int d = Y.n_rows;
  
  arma::cx_mat predict = zeros<cx_mat>(d,m);
  
  for(int i = 0; i < m; i++){
    
    R_CheckUserInterrupt();
    
    predict.col(i) = FShape(test.col(i), train, Y, h);
    
  }
  
  return(predict);
  
}


// [[Rcpp::export]]
double GetShapeMSE(arma::cx_mat pred, arma::cx_mat val, arma::cx_vec center,
                   double radius){
  //Gets the MSE of a set of predictions from the true sphere values
  //on a sphere with the given radius and center. Used by the 
  //cross validation function. Uses the norm of kendall's shape
  //space.
  //
  //Args:
  // pred: arma::cx_mat d x m. Predicted values.
  // val: arma::cx_mat d x m. True/validation values.
  // center: arma::cx_vec d. Center of sphere
  // radius: double. Radius of sphere.
  //
  //Returns:
  // mse: double. MSE of the values.
  /////////////////////////////////////////////////////////////////////////////
  
  double mse_ret = 0;
  
  int d = center.n_elem;
  
  int m = pred.n_cols;
  
  arma::cx_vec pred_normal(d);
  
  arma::cx_vec val_normal(d);
  
  for(int i = 0; i < m; i++){
    
    pred_normal = pred.col(i) - center;
    
    pred_normal = normalise(pred_normal);
    
    val_normal= val.col(i) - center;
    
    val_normal = normalise(val_normal);
    
    arma::cx_vec diff = KendallsLogMap_x(pred_normal, val_normal);
    
    arma::vec complex_norms = square(real(diff)) + square(imag(diff));

    mse_ret += sqrt(accu(complex_norms));
    
  }
  
  return(mse_ret);
  
}


// [[Rcpp::export]]
Rcpp::List IntrinsicShapeCrossValPredict(arma::mat x, arma::mat X, arma::cx_mat Y,
                                          arma::cx_vec center, double radius, double eps = .001,
                                          double h_low = .01, double h_high = 2,
                                          double h_by = .1, int k = 10, 
                                          bool talk = false, double step = .01, int max_iter = 100000){
  //Performs cross validation selection of the bandwidth for extrinsic kernel
  //regression, and then predicts the sphere response for a given test set of
  //features. Uses the inclusion map for the sphere, geodesic distance for
  //residuals.
  //
  //Args:
  // x: arma::mat p x m. Test matrix of features for the prediction
  // X: arma::mat p x n. Training data set of features.
  // Y: arma::cx_mat d x n. Training set of sphere-valued responses
  //    corresponding to the training set.
  // h_low: double. The lower bound for h to be tried for bandwidth selection.
  //        DEFAULT is .01.
  // h_high: double. The upper bound for h to be tried for bandwidth selection.
  //         DEFAULT is 2.
  // h_by: double. The intervals that the bandwidth should be tried at over
  //       the range. DEFAULT is .1.
  // k:    int. The number of cross validation sets to use.
  //
  //Returns:
  // prediction: arma::cx_mat  d x m. The predicted sphere response for the test
  //             observations contained in x.
  // h: double. The selected bandwidth from cross validation used to make the
  //    prediction.
  // MSE: arma::mat. The MSE from all the cross validation trials.
  // h_seq: arma::vec. The grid of values used in the cross validation 
  //        search.
  /////////////////////////////////////////////////////////////////////////////
  
  
  int n = X.n_cols;
  
  //int p = X.n_rows;
  
  //Get the number of observations to go into each matrix
  
  int n_cv = (int)n / k;
  
  //int rem_cv = n % k;
  
  //Initialize the indexing list
  
  arma::uvec obs_ind((n_cv*k));
  
  for(int i = 0; i < (n_cv * k); i++){
    
    obs_ind(i) = i;
    
  }
  
  obs_ind = shuffle(obs_ind);
  
  //Initialize the bandwidth sequence
  
  int hsteps = (h_high - h_low) / h_by + 2;
  
  arma::vec h_seq(hsteps);
  
  for(int ii = 0; ii < hsteps; ii++){
    
    h_seq(ii) = h_low + ii*h_by;
    
  }
  
  //initialize the MSE matrix
  
  arma::mat mse_mat(k, hsteps);
  
  
  //For each of the bandwidth values Find the MSE using each of the 
  //k subsamples as the validation set
  arma::mat cv_Xval;
  
  arma::mat cv_Xtrain;
  
  arma::cx_mat cv_Yval;
  
  arma::cx_mat cv_Ytrain;
  //Pick a validation
  for(int kk = 0; kk < k; kk++){
    
    if(talk){
      Rcpp::Rcout << "Checking Validation Group " << kk + 1 << std::endl;
    }
    
    //Get the subset
    if(kk == (k-1)){
      
      uvec cv_val_ind = SubVec(obs_ind, n_cv * (kk), n - 1);
      
      uvec cv_train_ind = AntiSubVec(obs_ind, n_cv * (kk), n - 1);
      
      cv_Xval = X.cols(cv_val_ind);
      
      cv_Xtrain = X.cols(cv_train_ind);
      
      cv_Yval = Y.cols(cv_val_ind);
      
      cv_Ytrain = Y.cols(cv_train_ind);
      
    } else {
      
      uvec cv_val_ind = SubVec(obs_ind, n_cv * kk, n_cv * (kk + 1) - 1);
      
      uvec cv_train_ind = AntiSubVec(obs_ind, n_cv * kk, n_cv * (kk + 1) - 1);
      
      cv_Xval = X.cols(cv_val_ind);
      
      cv_Xtrain = X.cols(cv_train_ind);
      
      cv_Yval = Y.cols(cv_val_ind);
      
      cv_Ytrain = Y.cols(cv_train_ind);
      
    }
    
    //For the validation set, check each of the bandwidth values
    
    for(int hh = 0; hh < hsteps; hh++){
      
      R_CheckUserInterrupt();
      
      //get the predicted values, will be d x (training set size)
        
      
      arma::cx_mat val_predict = IntrinsicPredictShape(cv_Xval, cv_Xtrain, cv_Ytrain,
                                               center, radius, eps, h_seq(hh),
                                               step, max_iter);
      
      mse_mat(kk, hh) = GetShapeMSE(val_predict, cv_Yval, center, radius);
      
      
    }
    
  }
  
  //Select a bandwidth value based on the performance of each
  
  arma::rowvec total_mse = sum(mse_mat);
  
  arma::uvec band_ind = find(total_mse == min(total_mse));
  
  double h = h_seq(band_ind(0));
  
  arma::cx_mat prediction = IntrinsicPredictShape(x, X, Y, center, radius, eps, h,
                                                  step, max_iter);
  
  return(Rcpp::List::create(Rcpp::Named("prediction") = prediction,
                            Rcpp::Named("h") = h,
                            Rcpp::Named("MSE") = mse_mat,
                            Rcpp::Named("hseq") = h_seq));
  
  
}



// [[Rcpp::export]]
Rcpp::List FShapeCrossValPredict(arma::mat x, arma::mat X, arma::cx_mat Y,
                                 arma::cx_vec center, double radius,
                                 double h_low = .01, double h_high = 2,
                                 double h_by = .1, int k = 10, 
                                 bool talk = false){
  //Performs cross validation selection of the bandwidth for extrinsic kernel
  //regression, and then predicts the sphere response for a given test set of
  //features. Uses the inclusion map for the sphere, geodesic distance for
  //residuals.
  //
  //Args:
  // x: arma::mat p x m. Test matrix of features for the prediction
  // X: arma::mat p x n. Training data set of features.
  // Y: arma::cx_mat d x n. Training set of sphere-valued responses
  //    corresponding to the training set.
  // h_low: double. The lower bound for h to be tried for bandwidth selection.
  //        DEFAULT is .01.
  // h_high: double. The upper bound for h to be tried for bandwidth selection.
  //         DEFAULT is 2.
  // h_by: double. The intervals that the bandwidth should be tried at over
  //       the range. DEFAULT is .1.
  // k:    int. The number of cross validation sets to use.
  //
  //Returns:
  // prediction: arma::cx_mat  d x m. The predicted sphere response for the test
  //             observations contained in x.
  // h: double. The selected bandwidth from cross validation used to make the
  //    prediction.
  // MSE: arma::mat. The MSE from all the cross validation trials.
  // h_seq: arma::vec. The grid of values used in the cross validation 
  //        search.
  /////////////////////////////////////////////////////////////////////////////
  if(talk){
    Rcpp::Rcout << "Start" << std::endl;
  }
  
  int n = X.n_cols;
  
  //int p = X.n_rows;
  
  //Get the number of observations to go into each matrix
  
  int n_cv = (int)n / k;
  
  //int rem_cv = n % k;
  
  //Initialize the indexing list
  if(talk){
    Rcpp::Rcout << "Initialize index lists" << std::endl;
  }
  
  arma::uvec obs_ind((n_cv*k));
  
  for(int i = 0; i < (n_cv * k); i++){
    
    obs_ind(i) = i;
    
  }
  
  obs_ind = shuffle(obs_ind);
  
  //Initialize the bandwidth sequence
  if(talk){
    Rcpp::Rcout << "Initialize bandwidth sequence" << std::endl;
  }
  
  int hsteps = (h_high - h_low) / h_by + 2;
  
  arma::vec h_seq(hsteps);
  
  for(int ii = 0; ii < hsteps; ii++){
    
    h_seq(ii) = h_low + ii*h_by;
    
  }
  
  //initialize the MSE matrix
  
  arma::mat mse_mat(k, hsteps);
  
  
  //For each of the bandwidth values Find the MSE using each of the 
  //k subsamples as the validation set
  arma::mat cv_Xval;
  
  arma::mat cv_Xtrain;
  
  arma::cx_mat cv_Yval;
  
  arma::cx_mat cv_Ytrain;
  //Pick a validation
  for(int kk = 0; kk < k; kk++){
    
    if(talk){
      Rcpp::Rcout << "Checking Validation Group " << kk + 1 << std::endl;
    }
    
    //Get the subset
    if(kk == (k-1)){
      
      uvec cv_val_ind = SubVec(obs_ind, n_cv * (kk), n - 1);
      
      uvec cv_train_ind = AntiSubVec(obs_ind, n_cv * (kk), n - 1);
      
      cv_Xval = X.cols(cv_val_ind);
      
      cv_Xtrain = X.cols(cv_train_ind);
      
      cv_Yval = Y.cols(cv_val_ind);
      
      cv_Ytrain = Y.cols(cv_train_ind);
      
    } else {
      
      uvec cv_val_ind = SubVec(obs_ind, n_cv * kk, n_cv * (kk + 1) - 1);
      
      uvec cv_train_ind = AntiSubVec(obs_ind, n_cv * kk, n_cv * (kk + 1) - 1);
      
      cv_Xval = X.cols(cv_val_ind);
      
      cv_Xtrain = X.cols(cv_train_ind);
      
      cv_Yval = Y.cols(cv_val_ind);
      
      cv_Ytrain = Y.cols(cv_train_ind);
      
    }
    
    //For the validation set, check each of the bandwidth values
    
    for(int hh = 0; hh < hsteps; hh++){
      
      R_CheckUserInterrupt();
      
      //get the predicted values, will be d x (training set size)

      arma::cx_mat val_predict = FPredictShape(cv_Xval, cv_Xtrain, cv_Ytrain,
                                               center, radius, h_seq(hh));
    
    
      
      mse_mat(kk, hh) = GetShapeMSE(val_predict, cv_Yval, center, radius);
      
      
    }
    
  }
  
  //Select a bandwidth value based on the performance of each
  
  if(talk){
    Rcpp::Rcout << "Exiting fitting and training" << std::endl;
  }
  
  arma::rowvec total_mse = sum(mse_mat);
  
  if(talk){
    Rcpp::Rcout << "Select bandwidth" << std::endl;
  }
  
  arma::uvec band_ind = find(total_mse == min(total_mse));

  double h = h_seq(band_ind(0));
  
  if(talk){
    Rcpp::Rcout << "Fit with optimal bandwidth on full data" << std::endl;
  }
  
  arma::cx_mat prediction = FPredictShape(x, X, Y, center, radius, h);
  
  if(talk){
    Rcpp::Rcout << "Return" << std::endl;
  }
  
  return(Rcpp::List::create(Rcpp::Named("prediction") = prediction,
                            Rcpp::Named("h") = h,
                            Rcpp::Named("MSE") = mse_mat,
                            Rcpp::Named("hseq") = h_seq));
  
  
}
