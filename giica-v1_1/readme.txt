This GI-ICA project is a Matlab implementation of the gradient
iteration based algorithms for Independent Component Analysis designed
to be robust to Gaussian noise as described in the papers:

Fast Algorithms for Gaussian Noise Invariant Independent Component Analysis
by: James Voss, Luis Rademacher, and Mikhail Belkin
NIPS 2013

A Pseudo-Euclidean Iteration for Optimal Recovery in Noisy ICA
by: James Voss, Mikhail Belkin, and Luis Rademacher
NIPS 2015

This code is made available under the GPLv3 license.  We ask that any
academic work which uses this code cite the most relevant of the
aforementioned papers.

Usage of code: 
The code should be kept together in a single directory in order to
satisfy all interdependencies.  The file demo_giica.m provides a basic
demo showing how to run the ICA algorithms on simulated data.  For
running the ICA algorithms on your own data, the main function to
interface with is in GIICA.m.  The algorithm is actually implemented
in the file ICA_Implementation.m along with a number of auxiliary
function files.

We refer the user to the comments at the top of the file GIICA.m for a
detailed explanation of all usage options.
