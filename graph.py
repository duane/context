import numpy
import pylab
import random
import math

def gaussian(x, sigma, mu):
  sigmaSquared = (sigma * sigma)
  lhs = 1.0 / math.sqrt(2.0 * math.pi * sigmaSquared)
  rhs = math.e ** (-(((x - mu)**2.0)/(2.0 * sigmaSquared)))
  return lhs * rhs

def chauvenet(data):
  sigma = numpy.std(data)
  mu = numpy.mean(data)
  n = len(data)
  evaluated = [(gaussian(x, sigma, mu) * n, x) for x in data]
  normal = [x for (p, x) in evaluated if p >= 0.5]
  outlier = [x for (p, x) in evaluated if p < 0.5]
  return (normal, outlier)

data = numpy.fromfile("output.dat", numpy.dtype(numpy.uint64))
if __name__ == "__main__":
  mean = numpy.mean(data)
  std = numpy.std(data)
 
  (normal, outlier) = chauvenet(data)

  print("Overall data:")
  print("\tMean: %f ns" % numpy.mean(data))
  print("\tStd: %f ns" % numpy.std(data))

  print("Normal data:")
  print("\tMean: %f ns" % numpy.mean(normal))
  print("\tStd: %f ns" % numpy.std(normal))

  print("Outliers:")
  print("\tMean (context switch estimate): %f ns" % numpy.mean(outlier))
  print("\tStd: %f ns" % numpy.std(outlier))

  pylab.subplot(311)
  pylab.plot(data)

  pylab.subplot(312)
  pylab.hist(normal, range=(20,40))

  pylab.subplot(313)
  pylab.plot(outlier)

  pylab.show()


# data = intervalTime :: x => t
# 
