// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// timesTwo
Rcpp::List timesTwo(int x);
RcppExport SEXP _RiemLong_timesTwo(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(timesTwo(x));
    return rcpp_result_gen;
END_RCPP
}
// GaussKern
double GaussKern(arma::vec x, arma::vec y, double h);
RcppExport SEXP _RiemLong_GaussKern(SEXP xSEXP, SEXP ySEXP, SEXP hSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type h(hSEXP);
    rcpp_result_gen = Rcpp::wrap(GaussKern(x, y, h));
    return rcpp_result_gen;
END_RCPP
}
// FEuc
arma::vec FEuc(arma::vec x, arma::mat X, arma::mat Y, double h);
RcppExport SEXP _RiemLong_FEuc(SEXP xSEXP, SEXP XSEXP, SEXP YSEXP, SEXP hSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< double >::type h(hSEXP);
    rcpp_result_gen = Rcpp::wrap(FEuc(x, X, Y, h));
    return rcpp_result_gen;
END_RCPP
}
// FEuc2
arma::mat FEuc2(arma::vec x, arma::mat X, arma::mat Y, double h);
RcppExport SEXP _RiemLong_FEuc2(SEXP xSEXP, SEXP XSEXP, SEXP YSEXP, SEXP hSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< double >::type h(hSEXP);
    rcpp_result_gen = Rcpp::wrap(FEuc2(x, X, Y, h));
    return rcpp_result_gen;
END_RCPP
}
// FSphere
arma::vec FSphere(arma::vec x, arma::mat X, arma::mat Y, arma::vec center, double h, double radius);
RcppExport SEXP _RiemLong_FSphere(SEXP xSEXP, SEXP XSEXP, SEXP YSEXP, SEXP centerSEXP, SEXP hSEXP, SEXP radiusSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type h(hSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    rcpp_result_gen = Rcpp::wrap(FSphere(x, X, Y, center, h, radius));
    return rcpp_result_gen;
END_RCPP
}
// SpherePredict
arma::mat SpherePredict(arma::mat test, arma::mat train, arma::mat Y, arma::vec center, double radius, double h);
RcppExport SEXP _RiemLong_SpherePredict(SEXP testSEXP, SEXP trainSEXP, SEXP YSEXP, SEXP centerSEXP, SEXP radiusSEXP, SEXP hSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type test(testSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type train(trainSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< double >::type h(hSEXP);
    rcpp_result_gen = Rcpp::wrap(SpherePredict(test, train, Y, center, radius, h));
    return rcpp_result_gen;
END_RCPP
}
// SphereDist
double SphereDist(arma::vec y1, arma::vec y2, arma::vec center, double radius);
RcppExport SEXP _RiemLong_SphereDist(SEXP y1SEXP, SEXP y2SEXP, SEXP centerSEXP, SEXP radiusSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y1(y1SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y2(y2SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    rcpp_result_gen = Rcpp::wrap(SphereDist(y1, y2, center, radius));
    return rcpp_result_gen;
END_RCPP
}
// GetSphereMSE
double GetSphereMSE(arma::mat pred, arma::mat val, arma::vec center, double radius);
RcppExport SEXP _RiemLong_GetSphereMSE(SEXP predSEXP, SEXP valSEXP, SEXP centerSEXP, SEXP radiusSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type pred(predSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type val(valSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    rcpp_result_gen = Rcpp::wrap(GetSphereMSE(pred, val, center, radius));
    return rcpp_result_gen;
END_RCPP
}
// SphereCrossValPredict
Rcpp::List SphereCrossValPredict(arma::mat x, arma::mat X, arma::mat Y, arma::vec center, double radius, double h_low, double h_high, double h_by, int k, bool talk);
RcppExport SEXP _RiemLong_SphereCrossValPredict(SEXP xSEXP, SEXP XSEXP, SEXP YSEXP, SEXP centerSEXP, SEXP radiusSEXP, SEXP h_lowSEXP, SEXP h_highSEXP, SEXP h_bySEXP, SEXP kSEXP, SEXP talkSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< double >::type h_low(h_lowSEXP);
    Rcpp::traits::input_parameter< double >::type h_high(h_highSEXP);
    Rcpp::traits::input_parameter< double >::type h_by(h_bySEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< bool >::type talk(talkSEXP);
    rcpp_result_gen = Rcpp::wrap(SphereCrossValPredict(x, X, Y, center, radius, h_low, h_high, h_by, k, talk));
    return rcpp_result_gen;
END_RCPP
}
// SphereCrossValPredictQuiet
arma::mat SphereCrossValPredictQuiet(arma::mat x, arma::mat X, arma::mat Y, arma::vec center, double radius, double h_low, double h_high, double h_by, int k);
RcppExport SEXP _RiemLong_SphereCrossValPredictQuiet(SEXP xSEXP, SEXP XSEXP, SEXP YSEXP, SEXP centerSEXP, SEXP radiusSEXP, SEXP h_lowSEXP, SEXP h_highSEXP, SEXP h_bySEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< double >::type h_low(h_lowSEXP);
    Rcpp::traits::input_parameter< double >::type h_high(h_highSEXP);
    Rcpp::traits::input_parameter< double >::type h_by(h_bySEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    rcpp_result_gen = Rcpp::wrap(SphereCrossValPredictQuiet(x, X, Y, center, radius, h_low, h_high, h_by, k));
    return rcpp_result_gen;
END_RCPP
}
// IntrinsicSphere
arma::vec IntrinsicSphere(arma::vec x, arma::mat X, arma::mat Y, arma::vec center, double radius, double eps, double h, double step, int max_iter);
RcppExport SEXP _RiemLong_IntrinsicSphere(SEXP xSEXP, SEXP XSEXP, SEXP YSEXP, SEXP centerSEXP, SEXP radiusSEXP, SEXP epsSEXP, SEXP hSEXP, SEXP stepSEXP, SEXP max_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< double >::type h(hSEXP);
    Rcpp::traits::input_parameter< double >::type step(stepSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(IntrinsicSphere(x, X, Y, center, radius, eps, h, step, max_iter));
    return rcpp_result_gen;
END_RCPP
}
// IntrinsicPredict
arma::mat IntrinsicPredict(arma::mat test, arma::mat train, arma::mat Y, arma::vec center, double radius, double eps, double h, double step, int max_iter);
RcppExport SEXP _RiemLong_IntrinsicPredict(SEXP testSEXP, SEXP trainSEXP, SEXP YSEXP, SEXP centerSEXP, SEXP radiusSEXP, SEXP epsSEXP, SEXP hSEXP, SEXP stepSEXP, SEXP max_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type test(testSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type train(trainSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< double >::type h(hSEXP);
    Rcpp::traits::input_parameter< double >::type step(stepSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(IntrinsicPredict(test, train, Y, center, radius, eps, h, step, max_iter));
    return rcpp_result_gen;
END_RCPP
}
// FSubspace
Rcpp::List FSubspace(arma::vec x, arma::mat X, arma::mat Y, double h);
RcppExport SEXP _RiemLong_FSubspace(SEXP xSEXP, SEXP XSEXP, SEXP YSEXP, SEXP hSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< double >::type h(hSEXP);
    rcpp_result_gen = Rcpp::wrap(FSubspace(x, X, Y, h));
    return rcpp_result_gen;
END_RCPP
}
// ConwaySphereDistance
double ConwaySphereDistance(arma::mat p1, arma::mat p2, int d);
RcppExport SEXP _RiemLong_ConwaySphereDistance(SEXP p1SEXP, SEXP p2SEXP, SEXP dSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type p1(p1SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type p2(p2SEXP);
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    rcpp_result_gen = Rcpp::wrap(ConwaySphereDistance(p1, p2, d));
    return rcpp_result_gen;
END_RCPP
}
// ConwayLowerSphereDistance
double ConwayLowerSphereDistance(arma::mat p1, arma::mat p2, int d, int k);
RcppExport SEXP _RiemLong_ConwayLowerSphereDistance(SEXP p1SEXP, SEXP p2SEXP, SEXP dSEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type p1(p1SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type p2(p2SEXP);
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    rcpp_result_gen = Rcpp::wrap(ConwayLowerSphereDistance(p1, p2, d, k));
    return rcpp_result_gen;
END_RCPP
}
// SubspacePairDists
arma::mat SubspacePairDists(arma::mat x, int n, int d);
RcppExport SEXP _RiemLong_SubspacePairDists(SEXP xSEXP, SEXP nSEXP, SEXP dSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    rcpp_result_gen = Rcpp::wrap(SubspacePairDists(x, n, d));
    return rcpp_result_gen;
END_RCPP
}
// IntrinsicSphereCrossValPredict
Rcpp::List IntrinsicSphereCrossValPredict(arma::mat x, arma::mat X, arma::mat Y, arma::vec center, double radius, double h_low, double h_high, double h_by, int k, bool talk);
RcppExport SEXP _RiemLong_IntrinsicSphereCrossValPredict(SEXP xSEXP, SEXP XSEXP, SEXP YSEXP, SEXP centerSEXP, SEXP radiusSEXP, SEXP h_lowSEXP, SEXP h_highSEXP, SEXP h_bySEXP, SEXP kSEXP, SEXP talkSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< double >::type h_low(h_lowSEXP);
    Rcpp::traits::input_parameter< double >::type h_high(h_highSEXP);
    Rcpp::traits::input_parameter< double >::type h_by(h_bySEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< bool >::type talk(talkSEXP);
    rcpp_result_gen = Rcpp::wrap(IntrinsicSphereCrossValPredict(x, X, Y, center, radius, h_low, h_high, h_by, k, talk));
    return rcpp_result_gen;
END_RCPP
}
// KendallsExponentialMap_x
arma::cx_vec KendallsExponentialMap_x(arma::cx_vec x, arma::cx_vec v);
RcppExport SEXP _RiemLong_KendallsExponentialMap_x(SEXP xSEXP, SEXP vSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cx_vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type v(vSEXP);
    rcpp_result_gen = Rcpp::wrap(KendallsExponentialMap_x(x, v));
    return rcpp_result_gen;
END_RCPP
}
// KendallsLogMap_x
arma::cx_vec KendallsLogMap_x(arma::cx_vec x, arma::cx_vec y);
RcppExport SEXP _RiemLong_KendallsLogMap_x(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cx_vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(KendallsLogMap_x(x, y));
    return rcpp_result_gen;
END_RCPP
}
// IntrinsicShape
arma::cx_vec IntrinsicShape(arma::vec x, arma::mat X, arma::cx_mat Y, arma::cx_vec center, double radius, double eps, double h, double step, int max_iter);
RcppExport SEXP _RiemLong_IntrinsicShape(SEXP xSEXP, SEXP XSEXP, SEXP YSEXP, SEXP centerSEXP, SEXP radiusSEXP, SEXP epsSEXP, SEXP hSEXP, SEXP stepSEXP, SEXP max_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::cx_mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< double >::type h(hSEXP);
    Rcpp::traits::input_parameter< double >::type step(stepSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(IntrinsicShape(x, X, Y, center, radius, eps, h, step, max_iter));
    return rcpp_result_gen;
END_RCPP
}
// IntrinsicPredictShape
arma::cx_mat IntrinsicPredictShape(arma::mat test, arma::mat train, arma::cx_mat Y, arma::cx_vec center, double radius, double eps, double h, double step, int max_iter);
RcppExport SEXP _RiemLong_IntrinsicPredictShape(SEXP testSEXP, SEXP trainSEXP, SEXP YSEXP, SEXP centerSEXP, SEXP radiusSEXP, SEXP epsSEXP, SEXP hSEXP, SEXP stepSEXP, SEXP max_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type test(testSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type train(trainSEXP);
    Rcpp::traits::input_parameter< arma::cx_mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< double >::type h(hSEXP);
    Rcpp::traits::input_parameter< double >::type step(stepSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(IntrinsicPredictShape(test, train, Y, center, radius, eps, h, step, max_iter));
    return rcpp_result_gen;
END_RCPP
}
// FShape
arma::cx_vec FShape(arma::vec x, arma::mat X, arma::cx_mat Y, double h);
RcppExport SEXP _RiemLong_FShape(SEXP xSEXP, SEXP XSEXP, SEXP YSEXP, SEXP hSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::cx_mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< double >::type h(hSEXP);
    rcpp_result_gen = Rcpp::wrap(FShape(x, X, Y, h));
    return rcpp_result_gen;
END_RCPP
}
// FPredictShape
arma::cx_mat FPredictShape(arma::mat test, arma::mat train, arma::cx_mat Y, arma::cx_vec center, double radius, double h);
RcppExport SEXP _RiemLong_FPredictShape(SEXP testSEXP, SEXP trainSEXP, SEXP YSEXP, SEXP centerSEXP, SEXP radiusSEXP, SEXP hSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type test(testSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type train(trainSEXP);
    Rcpp::traits::input_parameter< arma::cx_mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< double >::type h(hSEXP);
    rcpp_result_gen = Rcpp::wrap(FPredictShape(test, train, Y, center, radius, h));
    return rcpp_result_gen;
END_RCPP
}
// GetShapeMSE
double GetShapeMSE(arma::cx_mat pred, arma::cx_mat val, arma::cx_vec center, double radius);
RcppExport SEXP _RiemLong_GetShapeMSE(SEXP predSEXP, SEXP valSEXP, SEXP centerSEXP, SEXP radiusSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cx_mat >::type pred(predSEXP);
    Rcpp::traits::input_parameter< arma::cx_mat >::type val(valSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    rcpp_result_gen = Rcpp::wrap(GetShapeMSE(pred, val, center, radius));
    return rcpp_result_gen;
END_RCPP
}
// IntrinsicShapeCrossValPredict
Rcpp::List IntrinsicShapeCrossValPredict(arma::mat x, arma::mat X, arma::cx_mat Y, arma::cx_vec center, double radius, double eps, double h_low, double h_high, double h_by, int k, bool talk, double step, int max_iter);
RcppExport SEXP _RiemLong_IntrinsicShapeCrossValPredict(SEXP xSEXP, SEXP XSEXP, SEXP YSEXP, SEXP centerSEXP, SEXP radiusSEXP, SEXP epsSEXP, SEXP h_lowSEXP, SEXP h_highSEXP, SEXP h_bySEXP, SEXP kSEXP, SEXP talkSEXP, SEXP stepSEXP, SEXP max_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::cx_mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< double >::type h_low(h_lowSEXP);
    Rcpp::traits::input_parameter< double >::type h_high(h_highSEXP);
    Rcpp::traits::input_parameter< double >::type h_by(h_bySEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< bool >::type talk(talkSEXP);
    Rcpp::traits::input_parameter< double >::type step(stepSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(IntrinsicShapeCrossValPredict(x, X, Y, center, radius, eps, h_low, h_high, h_by, k, talk, step, max_iter));
    return rcpp_result_gen;
END_RCPP
}
// FShapeCrossValPredict
Rcpp::List FShapeCrossValPredict(arma::mat x, arma::mat X, arma::cx_mat Y, arma::cx_vec center, double radius, double h_low, double h_high, double h_by, int k, bool talk);
RcppExport SEXP _RiemLong_FShapeCrossValPredict(SEXP xSEXP, SEXP XSEXP, SEXP YSEXP, SEXP centerSEXP, SEXP radiusSEXP, SEXP h_lowSEXP, SEXP h_highSEXP, SEXP h_bySEXP, SEXP kSEXP, SEXP talkSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::cx_mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type center(centerSEXP);
    Rcpp::traits::input_parameter< double >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< double >::type h_low(h_lowSEXP);
    Rcpp::traits::input_parameter< double >::type h_high(h_highSEXP);
    Rcpp::traits::input_parameter< double >::type h_by(h_bySEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< bool >::type talk(talkSEXP);
    rcpp_result_gen = Rcpp::wrap(FShapeCrossValPredict(x, X, Y, center, radius, h_low, h_high, h_by, k, talk));
    return rcpp_result_gen;
END_RCPP
}
// loader
double loader(arma::mat X);
RcppExport SEXP _RiemLong_loader(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(loader(X));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_hello_world
arma::mat rcpparma_hello_world();
RcppExport SEXP _RiemLong_rcpparma_hello_world() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcpparma_hello_world());
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_outerproduct
arma::mat rcpparma_outerproduct(const arma::colvec& x);
RcppExport SEXP _RiemLong_rcpparma_outerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_outerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_innerproduct
double rcpparma_innerproduct(const arma::colvec& x);
RcppExport SEXP _RiemLong_rcpparma_innerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_innerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_bothproducts
Rcpp::List rcpparma_bothproducts(const arma::colvec& x);
RcppExport SEXP _RiemLong_rcpparma_bothproducts(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_bothproducts(x));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RiemLong_timesTwo", (DL_FUNC) &_RiemLong_timesTwo, 1},
    {"_RiemLong_GaussKern", (DL_FUNC) &_RiemLong_GaussKern, 3},
    {"_RiemLong_FEuc", (DL_FUNC) &_RiemLong_FEuc, 4},
    {"_RiemLong_FEuc2", (DL_FUNC) &_RiemLong_FEuc2, 4},
    {"_RiemLong_FSphere", (DL_FUNC) &_RiemLong_FSphere, 6},
    {"_RiemLong_SpherePredict", (DL_FUNC) &_RiemLong_SpherePredict, 6},
    {"_RiemLong_SphereDist", (DL_FUNC) &_RiemLong_SphereDist, 4},
    {"_RiemLong_GetSphereMSE", (DL_FUNC) &_RiemLong_GetSphereMSE, 4},
    {"_RiemLong_SphereCrossValPredict", (DL_FUNC) &_RiemLong_SphereCrossValPredict, 10},
    {"_RiemLong_SphereCrossValPredictQuiet", (DL_FUNC) &_RiemLong_SphereCrossValPredictQuiet, 9},
    {"_RiemLong_IntrinsicSphere", (DL_FUNC) &_RiemLong_IntrinsicSphere, 9},
    {"_RiemLong_IntrinsicPredict", (DL_FUNC) &_RiemLong_IntrinsicPredict, 9},
    {"_RiemLong_FSubspace", (DL_FUNC) &_RiemLong_FSubspace, 4},
    {"_RiemLong_ConwaySphereDistance", (DL_FUNC) &_RiemLong_ConwaySphereDistance, 3},
    {"_RiemLong_ConwayLowerSphereDistance", (DL_FUNC) &_RiemLong_ConwayLowerSphereDistance, 4},
    {"_RiemLong_SubspacePairDists", (DL_FUNC) &_RiemLong_SubspacePairDists, 3},
    {"_RiemLong_IntrinsicSphereCrossValPredict", (DL_FUNC) &_RiemLong_IntrinsicSphereCrossValPredict, 10},
    {"_RiemLong_KendallsExponentialMap_x", (DL_FUNC) &_RiemLong_KendallsExponentialMap_x, 2},
    {"_RiemLong_KendallsLogMap_x", (DL_FUNC) &_RiemLong_KendallsLogMap_x, 2},
    {"_RiemLong_IntrinsicShape", (DL_FUNC) &_RiemLong_IntrinsicShape, 9},
    {"_RiemLong_IntrinsicPredictShape", (DL_FUNC) &_RiemLong_IntrinsicPredictShape, 9},
    {"_RiemLong_FShape", (DL_FUNC) &_RiemLong_FShape, 4},
    {"_RiemLong_FPredictShape", (DL_FUNC) &_RiemLong_FPredictShape, 6},
    {"_RiemLong_GetShapeMSE", (DL_FUNC) &_RiemLong_GetShapeMSE, 4},
    {"_RiemLong_IntrinsicShapeCrossValPredict", (DL_FUNC) &_RiemLong_IntrinsicShapeCrossValPredict, 13},
    {"_RiemLong_FShapeCrossValPredict", (DL_FUNC) &_RiemLong_FShapeCrossValPredict, 10},
    {"_RiemLong_loader", (DL_FUNC) &_RiemLong_loader, 1},
    {"_RiemLong_rcpparma_hello_world", (DL_FUNC) &_RiemLong_rcpparma_hello_world, 0},
    {"_RiemLong_rcpparma_outerproduct", (DL_FUNC) &_RiemLong_rcpparma_outerproduct, 1},
    {"_RiemLong_rcpparma_innerproduct", (DL_FUNC) &_RiemLong_rcpparma_innerproduct, 1},
    {"_RiemLong_rcpparma_bothproducts", (DL_FUNC) &_RiemLong_rcpparma_bothproducts, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_RiemLong(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}