# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

timesTwo <- function(x) {
    .Call('_RiemLong_timesTwo', PACKAGE = 'RiemLong', x)
}

GaussKern <- function(x, y, h) {
    .Call('_RiemLong_GaussKern', PACKAGE = 'RiemLong', x, y, h)
}

FEuc <- function(x, X, Y, h = 1) {
    .Call('_RiemLong_FEuc', PACKAGE = 'RiemLong', x, X, Y, h)
}

FEuc2 <- function(x, X, Y, h = 1) {
    .Call('_RiemLong_FEuc2', PACKAGE = 'RiemLong', x, X, Y, h)
}

FSphere <- function(x, X, Y, center, h = 1, radius = 1) {
    .Call('_RiemLong_FSphere', PACKAGE = 'RiemLong', x, X, Y, center, h, radius)
}

SpherePredict <- function(test, train, Y, center, radius = 1, h = 1) {
    .Call('_RiemLong_SpherePredict', PACKAGE = 'RiemLong', test, train, Y, center, radius, h)
}

SphereDist <- function(y1, y2, center, radius) {
    .Call('_RiemLong_SphereDist', PACKAGE = 'RiemLong', y1, y2, center, radius)
}

GetSphereMSE <- function(pred, val, center, radius) {
    .Call('_RiemLong_GetSphereMSE', PACKAGE = 'RiemLong', pred, val, center, radius)
}

SphereCrossValPredict <- function(x, X, Y, center, radius, h_low = .01, h_high = 2, h_by = .1, k = 10L, talk = FALSE) {
    .Call('_RiemLong_SphereCrossValPredict', PACKAGE = 'RiemLong', x, X, Y, center, radius, h_low, h_high, h_by, k, talk)
}

SphereCrossValPredictQuiet <- function(x, X, Y, center, radius, h_low = .01, h_high = 2, h_by = .1, k = 10L) {
    .Call('_RiemLong_SphereCrossValPredictQuiet', PACKAGE = 'RiemLong', x, X, Y, center, radius, h_low, h_high, h_by, k)
}

IntrinsicSphere <- function(x, X, Y, center, radius, eps = .001, h = .6, step = .01, max_iter = 100000L) {
    .Call('_RiemLong_IntrinsicSphere', PACKAGE = 'RiemLong', x, X, Y, center, radius, eps, h, step, max_iter)
}

IntrinsicPredict <- function(test, train, Y, center, radius, eps = .001, h = .6, step = .01, max_iter = 100000L) {
    .Call('_RiemLong_IntrinsicPredict', PACKAGE = 'RiemLong', test, train, Y, center, radius, eps, h, step, max_iter)
}

FSubspace <- function(x, X, Y, h = 1) {
    .Call('_RiemLong_FSubspace', PACKAGE = 'RiemLong', x, X, Y, h)
}

ConwaySphereDistance <- function(p1, p2, d) {
    .Call('_RiemLong_ConwaySphereDistance', PACKAGE = 'RiemLong', p1, p2, d)
}

ConwayLowerSphereDistance <- function(p1, p2, d, k) {
    .Call('_RiemLong_ConwayLowerSphereDistance', PACKAGE = 'RiemLong', p1, p2, d, k)
}

SubspacePairDists <- function(x, n, d) {
    .Call('_RiemLong_SubspacePairDists', PACKAGE = 'RiemLong', x, n, d)
}

IntrinsicSphereCrossValPredict <- function(x, X, Y, center, radius, h_low = .01, h_high = 2, h_by = .1, k = 10L, talk = FALSE) {
    .Call('_RiemLong_IntrinsicSphereCrossValPredict', PACKAGE = 'RiemLong', x, X, Y, center, radius, h_low, h_high, h_by, k, talk)
}

KendallsExponentialMap_x <- function(x, v) {
    .Call('_RiemLong_KendallsExponentialMap_x', PACKAGE = 'RiemLong', x, v)
}

KendallsLogMap_x <- function(x, y) {
    .Call('_RiemLong_KendallsLogMap_x', PACKAGE = 'RiemLong', x, y)
}

IntrinsicShape <- function(x, X, Y, center, radius, eps = .001, h = .6, step = .01, max_iter = 100000L) {
    .Call('_RiemLong_IntrinsicShape', PACKAGE = 'RiemLong', x, X, Y, center, radius, eps, h, step, max_iter)
}

IntrinsicPredictShape <- function(test, train, Y, center, radius, eps = .001, h = .6, step = .01, max_iter = 100000L) {
    .Call('_RiemLong_IntrinsicPredictShape', PACKAGE = 'RiemLong', test, train, Y, center, radius, eps, h, step, max_iter)
}

FShape <- function(x, X, Y, h = 1) {
    .Call('_RiemLong_FShape', PACKAGE = 'RiemLong', x, X, Y, h)
}

FPredictShape <- function(test, train, Y, center, radius, h = .6) {
    .Call('_RiemLong_FPredictShape', PACKAGE = 'RiemLong', test, train, Y, center, radius, h)
}

GetShapeMSE <- function(pred, val, center, radius) {
    .Call('_RiemLong_GetShapeMSE', PACKAGE = 'RiemLong', pred, val, center, radius)
}

IntrinsicShapeCrossValPredict <- function(x, X, Y, center, radius, eps = .001, h_low = .01, h_high = 2, h_by = .1, k = 10L, talk = FALSE, step = .01, max_iter = 100000L) {
    .Call('_RiemLong_IntrinsicShapeCrossValPredict', PACKAGE = 'RiemLong', x, X, Y, center, radius, eps, h_low, h_high, h_by, k, talk, step, max_iter)
}

FShapeCrossValPredict <- function(x, X, Y, center, radius, h_low = .01, h_high = 2, h_by = .1, k = 10L, talk = FALSE) {
    .Call('_RiemLong_FShapeCrossValPredict', PACKAGE = 'RiemLong', x, X, Y, center, radius, h_low, h_high, h_by, k, talk)
}

loader <- function(X) {
    .Call('_RiemLong_loader', PACKAGE = 'RiemLong', X)
}

rcpparma_hello_world <- function() {
    .Call('_RiemLong_rcpparma_hello_world', PACKAGE = 'RiemLong')
}

rcpparma_outerproduct <- function(x) {
    .Call('_RiemLong_rcpparma_outerproduct', PACKAGE = 'RiemLong', x)
}

rcpparma_innerproduct <- function(x) {
    .Call('_RiemLong_rcpparma_innerproduct', PACKAGE = 'RiemLong', x)
}

rcpparma_bothproducts <- function(x) {
    .Call('_RiemLong_rcpparma_bothproducts', PACKAGE = 'RiemLong', x)
}

