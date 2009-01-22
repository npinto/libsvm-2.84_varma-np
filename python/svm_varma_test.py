#!/usr/bin/env python

from svm_varma import *
from scipy import *

# varma test


# -t 4
# => precomputed KERNEL

# -s {0,1,2,3,4} => C-SVC, nu-SVC etc.
# => constraint = -s 0! => C_SVC!

# -m 500 (500MB)

# -c parms.C

labels = [1,1,1,-1,-1,-1]
random.seed(1)
data = randn(6,6)
data[:len(labels)/2] += 1
kernel = dot(data.T,data)

samples = [[i+1]+list(kernel[i]) for i in xrange(len(kernel))]
print samples

problem = svm_problem(labels, samples);
varma_param = svm_parameter(svm_type=C_SVC, 
			    kernel_type=PRECOMPUTED, 
			    C=10, 
			    cache_size=500)
model = svm_model(problem, varma_param)

pred_label = model.predict(samples[0])
print pred_label

print "#"*80
print "obj:", model.obj
print "rho:", model.rho
print "total_sv:", model.total_sv
print "sv_coef:", model.sv_coef
print "SV:", model.SV
#print 

#print model.get_labels()

#print "obj:", model.obj
#print "rho:", model.rho
#print "nSV:", model.nSV
