Code title: "A Riemannian approach to low-rank algebraic Riccati equations"
(c) 2013-2014 Bamdev Mishra <b.mishra@ulg.ac.be>

This package contains a MATLAB implementation of the algorithm presented in the report.

B. Mishra and B. Vandereycken,
"A Riemannian approach to low-rank algebraic Riccati equations",
Technical report, arXiv:1312.4883, 2013.

This implementation is due to 
Bamdev Mishra <b.mishra@ulg.ac.be>, 2013.

The implementation is a research prototype still in development and is provided AS IS. 
No warranties or guarantees of any kind are given. Do not distribute this
code or use it other than for your own research without permission of the authors.

Feedback is greatly appreciated.

Version
-------------------------
- Code online on 22/02/2014.


Installation:
-------------------------
- Run "Install_mex.m". You do not need to do this step for subsequent usage. 
- Run "Run_me_first.m" to add folders to the working path. This needs to be done at the starting of each session.
- To check that everything works, run "Test.m" at Matlab command prompt.
  (you should see some plots at the end).

Files:
------
- Main_files/Riemannian_lowrank_riccati.m           -------- Main file containing the rank incrementing procedure
- Main_files/symfixedrankYYfactory_riccati.m        -------- Riemannian geometry for the proposed metric
                             

Disclaimer:
-----------

- All contents are written by Bamdev Mishra (b.mishra@ulg.ac.be) except,

- Manopt_1.0.4: the generic Matlab toolbox for optimizaton on manifolds. Downlaoded from http://Manopt.org

- Mex files used
  1) updateSparse.c is written by Stephen Becker, 11/10/08



