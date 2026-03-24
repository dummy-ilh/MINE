#---------------------------------------------------------
# File:   MIT18_05S22_class7-prep.r 
# Author: Jeremy Orloff
#
# MIT OpenCourseWare: https://ocw.mit.edu
# 18.05 Introduction to Probability and Statistics
# Spring 2022
# For information about citing these materials or our Terms of Use, visit:
# https://ocw.mit.edu/terms}.
#
#---------------------------------------------------------#
# This generates scatter plots  used in class7-prep

#----------------------
# Overlapping sums of uniform distributions
make_cor_pdf = FALSE  #SET TO TRUE TO OUTPUT pdf files
dot_radius = 0.5

run_one_cor = function(n, overlap, npts, col, make_pdf, pdf_path) {
    #X = sum of n uniform(0, 1)
    #Y = sum of n uniform(0, 1)
    #overlap = number in common between X and Y
    ncol = 2*n - overlap
    tmp = runif(npts*ncol, 0, 1)
    data = matrix(tmp, nrow=npts, ncol=ncol)  #raw data
  
    xcols = 1:n                         #x data = first n columns
    ycols = (n-overlap+1):(2*n-overlap) #y data overlaps x data in overlap columns
    if (n == 1) { #special case, only one column so rowSums barfs
        x = data[,xcols]
        y = data[,ycols]
    }
    else {
      x = rowSums(data[,xcols])
      y = rowSums(data[,ycols])
    }
    theory_cor = overlap/n       #Theoretical correlation
    sample_cor = cor(x,y)        #Sample correlation
    s = sprintf("(%d, %d) cor=%.2f, sample_cor=%.2f", n, overlap, theory_cor, sample_cor)

    if (make_pdf){
      pdf_width = 4
      pdf_height = 3.8
      pdf_name = sprintf("%s/class7-cor_%d_%d.pdf", pdf_path, n,overlap)
      print(pdf_name)
      pdf(file=pdf_name, width=pdf_width, height=pdf_height)
    }
    plot(x,y, col=col, cex=dot_radius, main=s, xlab="")
    title(xlab='x', mgp=c(2,1,0))
    abline(v=mean(x))
    abline(h=mean(y)) 
    if (make_pdf){
      dev.off()
    }
}

generate_overlapping_uniform = function(npts, make_cor_pdf, pdf_path) {
  run_one_cor(1,0,npts,rgb(0.2, 0.4, 0.9), make_cor_pdf, pdf_path)
  run_one_cor(2, 1, npts, 'red', make_cor_pdf, pdf_path)
  run_one_cor(5, 1, npts, 'orange', make_cor_pdf, pdf_path)
  run_one_cor(5, 3, npts, 'cyan', make_cor_pdf, pdf_path)
  run_one_cor(10, 5, npts, 'purple',make_cor_pdf, pdf_path)
  run_one_cor(10, 8, npts, 'magenta', make_cor_pdf, pdf_path)
}

#------------------------
# Bivariate normal scatter plots
run_one_bivariate_normal = function(rho, npts, col, make_pdf, pdf_path){
  #draw (u,v) independent standard normal
  u = rnorm(npts, 0, 1) 
  v = rnorm(npts, 0, 1)
  # manipulate to bivariate normal with means m1,m2, stdev s1, s2, cor rho
  m1=0; m2=0; s1=1; s2=1;
  x = m1 + s1*u
  y = m2 + s2*(rho*u + sqrt(1 - rho^2)*v)
  
  sample_cor = cor(x, y)            #Sample correlation
  s = sprintf("rho=%.2f, sample_cor=%.2f", rho, sample_cor)
  
  if (make_pdf){
    pdf_width = 4
    pdf_height = 3.8
    f = round(10*rho)
    pdf_name = sprintf("%s/class7-bivarnorm_%d.pdf", pdf_path, f)
    print(pdf_name)
    pdf(file=pdf_name, width=pdf_width, height=pdf_height)
  }
  plot(x,y, col=col, cex=dot_radius, main=s, xlab="")
  title(xlab='x', mgp=c(2,1,0))
  abline(v= mean(x))
  abline(h=mean(y)) 
  if (make_pdf){
    dev.off()
  }
}

generate_bivariate_normal_scatter_plots = function(npts, make_bivar_pdf, pdf_path) {
    run_one_bivariate_normal(0.0, npts, rgb(.2,.4,.9), make_bivar_pdf, pdf_path)
    run_one_bivariate_normal(0.3, npts, 'red', make_bivar_pdf, pdf_path)
    run_one_bivariate_normal(0.7, npts, 'orange', make_bivar_pdf, pdf_path)
    run_one_bivariate_normal(1.0, npts, 'cyan', make_bivar_pdf, pdf_path)
    run_one_bivariate_normal(-0.5, npts, 'purple', make_bivar_pdf, pdf_path)
    run_one_bivariate_normal(-0.9, npts, 'magenta', make_bivar_pdf, pdf_path)
}

####### UNCOMMENT TO MAKE SCATTER PLOTS
src_d = getSrcDirectory(run_one_cor)
setwd(src_d)
#pdf_path = "../../pdftex/img"
pdf_path = "."

unif_n_pts = 1000
make_unif_cor_pdf = FALSE
generate_overlapping_uniform(unif_n_pts, make_unif_cor_pdf, pdf_path)

bivar_n_pts = 1000
make_bivar_pdf = FALSE
generate_bivariate_normal_scatter_plots(bivar_n_pts, make_bivar_pdf, pdf_path)

