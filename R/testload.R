#' test function to load other packages
#' 
#' 
#' @export
test_load <- function(){
  X = matrix(rnorm(10*3),nrow=10)
  output = c(loader(X), norm(X))
  return(output)
}