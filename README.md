# Quantum Copy Channels

This repository contains various MATLAB programmes for the 
analysis and simulation of quantum cloners, as described 
in the article:

*Semidefinite Programming for Optimal Quantum Cloning: 
A Computational Framework'*


The programmes use the QETLAB library and the SDP solver SDPT3 with the 
CVX modelling framework.


### Content:

- **classes** <br/>
  Contains various cloners in the form of classes. 
  For clarity, a separate cloner is provided for each case $(M \mapsto N)$. 
  The directory also contains a general cloner class that covers most cases. 
  The cloners expect the states from which $\Omega$ can be formed as 
  input parameters.  
- **examples** <br/>
  Contains several examples
- **kraus_ops_validation** <br/>
  Algebraic Kraus operators were derived from the numerical results. 
  The directory contains programmes that validate the derived algebraic 
  Kraus operators.
- **numerical_result_tables** <br/>
  Contains the programmes used to calculate the numerical results (tables) 
  presented in the publication.
- **random_test** <br/>
  Contains various tests in which Kraus operators are generated and 
  then applied to a randomly generated input state. The fidelities of 
  the output are then calculated and compared with the theoretically 
  expected value.

Version 1.0 (May 2026)