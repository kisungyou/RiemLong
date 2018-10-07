#' Simulation for Lizhen's work
#' 
#' Kernel is defined as
#' \deqn{k(x,y) = a\exp \left( -b \|x-y\|_2^2 \right)}
#' 
#' @export
test.lizhen <- function(a=1,b=1,error=0.1,ntime=100,ncurve=10){
  # define mother function
  t = seq(from=-0.95,to=0.95,length.out = ntime)
  
}


#' @keywords internal
#' @noRd
test.lizhen.tangentialize <- function(X){
  Y = array(0,dim(X))
  for (i in 1:ncol(X)){
    tgt   = as.vector(X[,i])
    Y[,i] = tgt/sqrt(sum(tgt^2))
  }
  return(Y)
}