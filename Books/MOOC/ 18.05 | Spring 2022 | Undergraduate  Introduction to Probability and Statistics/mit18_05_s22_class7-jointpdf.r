#---------------------------------------------------------
# File:   MIT18_05S22_class7-jointPDF.r
# Author: Jeremy Orloff
#
# MIT OpenCourseWare: https://ocw.mit.edu
# 18.05 Introduction to Probability and Statistics
# Spring 2022
# For information about citing these materials or our Terms of Use, visit:
# https://ocw.mit.edu/terms}.
#
#---------------------------------------------------------#
# Class 7
# Code to generate random samples from
# the joint pdf f(x,y) = ax^n+by^n with a+b=n+1 
# on the unit square

# Marginal distributions and theory
#------
#   These are all easy to compute by direct 
#   integration
#   f_X(x) = ax^n + b/(n+1)
#   Density of Y given X:
#   f_(Y|X=x1) = f(y|x1) = f(x1,y)/f_X(x1) 
#              = (ax1^n+by^n)/(ax1^n+b/(n+1)) 
# CDFs 
#  F(x,y) = ax^(n+1)y/(n+1) + bxy^(n+1)/(n+1)
#  F_X(x) = ax^(n+1)/(n+1) + bx/(n+1)
#  F_(Y|X=x1) = (ax1^n*y + by^(n+1)/(n+1))/(ax1^n+b/(n+1)) 
#
# Univariate sample generation:
# If X has cdf F then generate x by
# 1. Generate u from uniform(0,1)
# 2. x = F^(-1)(u) (Assumes F is increasing, can patch up if F has some intervals where it's constant)
#
# Bivariate sample generation:
# 1. Generate x1 from F_X
# 2. Generate y1 from F_(Y|X=x1)
# 3. Return (x1,y1)

# Sample generation code
#-----
compute_b = function(n,a) {
  return (n+1 - a)
}
  
f = function(x, y, n, a) {
  # Joint pdf
  b = compute_b(n,a)
  return(a*x^n + b*y^n)
}

fX = function(x, n, a) {
  # Marginal pdf f_X(x)
  b = compute_b(n,a)
  return(a*x^n + b/(n+1))
}

fY_given_X = function(y, x1, n, a) {
  # Conditional pdf f_(Y|X=x1)
  b = compute_b(n,a)
  num = a*x1^n + b*y^n
  den = a*x1^n + b/(n+1)
  return(num/den)
}

F = function(x, y, n, a) {
  # Joint CDF
  return ( (x^(n+1)*y + x*y^(n+1))/(n+1) )
}

FX = function(x, n, a) {
  # Marginal CDF = F(x,1)
  b = compute_b(n, a)
  return ( a*x^(n+1)/(n+1) + b*x/(n+1) )
}

fY_given_X = function(y, x1, n, a) {
  # Conditional CDF F(y given X=x1)
  b = compute_b(n, a)
  num = a*x1^n*y + b*y^(n+1)/(n+1)
  den = a*x1^n + b/(n+1)
  return (num/den)
}

F_inverse_general = function(cdf, u, xmin, xmax, accuracy=12) {
  # Simple search for the inverse of a cdf
  # cdf is a cdf function: assumed to be increasing
  # The domain of cdf is assumed to be [xmin, xmax]
  # (More precisely we assume cdf^(-1)(u) is in
  # that interval.)
  # u is a list of values between 0 and 1
  # Finds cdf^(-1)(u) to within (xmax-xmin)/2^accuracy
  
  ulen = length(u);
  xret = 0*ulen
  for (i in 1:ulen) {
    x = -1;
    xmin_loop = xmin
    xmax_loop = xmax
    uval = u[i]
    for (j in 1:accuracy) {  
      x = (xmax_loop + xmin_loop)/2
      if (cdf(x) < uval) {
        xmin_loop = x
      }
      else {
        xmax_loop = x
      }
    }
    xret[i] = x
  }
  return(xret)
}

generate_samples = function(N, FXfunc, fY_given_Xfunc) {
  # N = number of samples
  # FXfunc is the CDF for X alone
  # fY_given_Xfunc is the CDF for Y given X)
  u = runif(N)
  x1 = F_inverse_general(FXfunc, u, 0, 1, accuracy=24)
  v = runif(N)
  y1 = 0*v
  for (i in 1:N) {
    vval = v[i]
    fY_given_X1 = function(y) {
      return( fY_given_Xfunc(y, x1[i]) )
    }
    y1[i] = F_inverse_general(fY_given_X1, vval, 0, 1, 
                        accuracy=24)    
  }
  ret = cbind(x1,y1)
  colnames(ret) = NULL
  return(ret)
}

TEST_FINVERSE = function(n, a) {
  FTest = function(x) {
    return (FX(x, n, a))
  }
  x = seq(0,1,.001)
  plot(x, FTest(x), type='l', col='red', lwd=2)
  u = seq(0,1,.001)
  x = F_inverse_general(FTest, u, 0, 1, accuracy=24)
  # Should be identical to red plot
  lines(x,u, col='blue', lwd=1)
}

output1 = function(N, n, a, do_scatter_plot=FALSE) {
  # N = number of samples
  # n = n in pdf
  # a = a in pdf
  FXfunc = function(x) {
    return (FX(x, n, a)) 
  }
  FYGXfunc = function(y,x1){ 
    return(fY_given_X(y, x1, n, a))
  }
  b = compute_b(n, a)
  smps = generate_samples(N, FXfunc , FYGXfunc)
  x = smps[,1]
  y = smps[,2]
  cv = cov(x,y)
  cat('Est. Covariance:', cv, '\n')
  
  cat("a, b, n:", a, b, n, '\n')
  mx = a/(n+2) + b/(2*(n+1))
  my = a/(2*(n+1)) + b/(n+2)
  mxx = a/(n+3) + b/(3*(n+1))
  myy = a/(3*(n+1)) + b/(n+3)
  mxy = (a+b)/(2*n+4)
  sdx_true = sqrt(mxx -mx^2)
  sdy_true = sqrt(myy-my^2)
  cxy_true = mxy - mx*my   #Covariance
  rhoxy_true = cxy_true/(sdx_true*sdy_true)
  cat('True std devs:', sdx_true, sdy_true, '\n')
  cat('True cov, cor:', cxy_true, rhoxy_true, '\n')
  sdx_est = sd(x)
  sdy_est = sd(y)
  cxy_est = cov(x,y)
  rho_est = cor(x,y) # = cxy_est/(sdx_est*sdy_est)
  cat('Estimated std devs:', sdx_est, sdy_est, '\n')
  cat('Estimated cov, cor:', cxy_est, rho_est, '\n')

  if (do_scatter_plot) {
    par('mar'= c(4,3,1,1)+0.1, mgp=c(2,1,0))
    plot(smps, pch=19, col='blue', cex=0.5,
         xlab='x', ylab='y')
    abline(h=0.5, col='black', lwd=1)
    abline(v=0.5, col='black', lwd=1)
    
    if (FALSE) {
      # linear regression is not really the right
      # thing to show the relationship between
      # x and yfor this problem 
      lmfit = lm(y ~ x)
      abline(lmfit, col='orange', lwd=4)
      print(lmfit)
    }
  }
}

#----
a=2; n=3
#output1(10000, n, a)
# Export pdf from GUI
output1(2000, n, a, do_scatter_plot=T)
a = 10; n=19
output1(1000, n, a, do_scatter_plot=T)
# TEST_FINVERSE(2,0)

