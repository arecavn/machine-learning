# ----------
#
# In this exercise, you will put the finishing touches on a perceptron class.
#
# Finish writing the activate() method by using np.dot to compute signal
# strength and then add in a threshold for perceptron activation.
#
# ----------


import numpy as np


class Perceptron:
    """
    This class models an artificial neuron with step activation function.
    """

    def __init__(self, weights = np.array([1]), threshold = 0):
        """
        Initialize weights and threshold based on input arguments. Note that no
        type-checking is being performed here for simplicity.
        """
        self.weights = weights
        self.threshold = threshold

    def activate(self,inputs):
        """
        Takes in @param inputs, a list of numbers equal to length of weights.
        @return the output of a threshold perceptron with given inputs based on
        perceptron weights and threshold.
        """

        # INSERT YOUR CODE HERE

        # TODO: calculate the strength with which the perceptron fires
        totalsum = np.dot(self.weights,inputs)
        print "totalsum: ", totalsum, " threshold: ", self.threshold

        # TODO: return 0 or 1 based on the threshold
        # Note that here, and the rest of the mini-project, that signal strength equal to the threshold results in a 0 being output (rather than 1).
        # It is required that the dot product be strictly greater than the threshold, rather than greater than or equal to the threshold, to pass the assertion tests.
        result = (totalsum > self.threshold)
        return result


def test():
    """
    A few tests to make sure that the perceptron class performs as expected.
    Nothing should show up in the output if all the assertions pass.
    """
    p1 = Perceptron(np.array([1, 2]), 0.)
    assert p1.activate(np.array([ 1,-1])) == 0 # < threshold --> 0
    assert p1.activate(np.array([-1, 1])) == 1 # > threshold --> 1
    assert p1.activate(np.array([ 2,-1])) == 0 # on threshold --> 0

if __name__ == "__main__":
    test()