#
#
# Regression and Classification programming exercises
#
#


#
#	In this exercise we will be taking a small data set and computing a linear function
#	that fits it, by hand.
#

#	the data set

import numpy as np

sleep = [5,6,7,8,10]
scores = [65,51,75,75,86]


def compute_regression(sleep,scores):

    #	First, compute the average amount of each list

    avg_sleep = np.mean(sleep)
    avg_scores = np.mean(scores)

    #	Then normalize the lists by subtracting the mean value from each entry

    normalized_sleep = sleep - avg_sleep
    normalized_scores = scores - avg_scores

    # print sum(normalized_scores * normalized_sleep)
    #	Compute the slope of the line by taking     the sum over each student
    #	of the product of their normalized sleep times their normalized test score.
    #	Then divide this by the sum of squares of the normalized sleep times.

    #print np.sqrt(np.abs(normalized_sleep))
    slope = np.sum((normalized_sleep * normalized_scores))/np.sum(normalized_sleep**2)

    # try polyfit https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.polyfit.html
    # (x,y,degree)
    s2 = np.polyfit(sleep, scores,1)
    print s2

    #	Finally, We have a linear function of the form
    #	y - avg_y = slope * ( x - avg_x )
    #	Rewrite this function in the form
    #	y = m * x + b
    #	Then return the values m, b
    # scores - avg_scores = slope * (sleep - avg_sleep) -> scores = slope * (sleep - avg_sleep) + avg_scores
    # -> scores = slope*sleep + (- slope * avg_sleep) + avg_scores

    m = slope
    b = avg_scores - (slope * avg_sleep)

    return m,b


if __name__=="__main__":
    m,b = compute_regression(sleep,scores)
    print "Your linear model is y={}*x+{}".format(m,b)
    print (b+ (0.0764431189382*m))

# Testing a random point: 0.0764431189382
    # 81.6
    # Your regression returns y = 51.5520699739
    # The expected value of y is 31.1212015853