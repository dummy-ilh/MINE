#---------------------------------------------------------
# File:   MIT18_05S22_class6-prep-b.r 
# Author: Jeremy Orloff
#
# MIT OpenCourseWare: https://ocw.mit.edu
# 18.05 Introduction to Probability and Statistics
# Spring 2022
# For information about citing these materials or our Terms of Use, visit:
# https://ocw.mit.edu/terms}.
#
#---------------------------------------------------------
do_example_1 = TRUE
do_hist_examples = TRUE

#---------------------------------
if (do_example_1) {
  # Example 1: Averages of Bernoulli variables
  # The number of heads in n flips of a fair coin is
  # modeled by binomial(n, 0.5)
  # We use the cdf function pbinom(x, n, 0.5) to compute
  # probabilities for various n

  # Probability of 4 to 6 heads in 10 flips
  # Note:  since pbinom(x, 10, 0.5) is P(X <= x) we use
  # pbinom(3, 10, 0.5) so X=4 is included
  # in the probability
  p = pbinom(6, 10, 0.5) - pbinom(3, 10, 0.5)
  cat(p, '\n') # 0.65625

  # We look at the probability of between 40 and 60% heads
  # for n = 50, 100, 500, 1000
  p = pbinom(30, 50, 0.5) - pbinom(19, 50, 0.5)
  cat(p, '\n') # 0.8810795
  p = pbinom(60, 100, 0.5) - pbinom(39, 100, 0.5)
  cat(p, '\n') # 0.9647998
  p = pbinom(300, 500, 0.5) - pbinom(199, 500, 0.5)
  cat(p, '\n') # 0.9999941
  p = pbinom(600, 1000, 0.5) - pbinom(399, 1000, 0.5)
  cat(p, '\n') # 1

  # Next we look at the probability of between 49 and 51 percent heads
  # (again we include the endpoints in the probability)

  # We look at the probability of between 40 and 60% heads for
  # n = 50, 100, 500, 1000
  p = pbinom(5, 10, 0.5) - pbinom(4, 10, 0.5)
  cat(p, '\n') # 0.2460937
  p = pbinom(51, 100, 0.5) - pbinom(48, 100, 0.5)
  cat(p, '\n') # 0.2356466
  p = pbinom(510, 1000, 0.5) - pbinom(489, 1000, 0.5)
  cat(p, '\n') # 0.49334
  p = pbinom(5100, 10000, 0.5) - pbinom(4899, 10000, 0.5)
  cat(p, '\n') # 0.9555742
}
#---------------------------------
if (do_hist_examples) {
  # Histogram examples
  x = c(1,2,2,3,3,3,4,4,4,4)/2
  brks = c(0.25, 0.75, 1.25, 1.75, 2.25)
  hist(x, breaks=brks, col="purple", freq=TRUE)
  hist(x, breaks=brks, col="magenta", freq=FALSE)

  brks = c(0,1,2,3,4)/2
  hist(x,breaks=brks, col="red", freq=TRUE)
  hist(x,breaks=brks, col="orange", freq=FALSE)

  brks = c(0,1,2)
  hist(x,breaks=brks, col="blue", freq=FALSE)

  brks = c(0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25)/2
  hist(x,breaks=brks, col="cyan", freq=FALSE)

  #unequal size bins
  cat("We expect using freq=TRUE with unequal bin sizes will generate a warning.\n")
  brks = c(0,1,3,4)/2
  hist(x,breaks=brks, col="red", freq=TRUE) #GENERATES WARNING
  hist(x,breaks=brks, col="orange", freq=FALSE)
}
