This repository is for a MLB Fantasy Sports Project that contains 3 roughly parts:

1. Scrape reference data using the Selenium and BeautifulSoup packages in Python

2. Train an xboost model to predict player scores

3. Run a multiple constraints knapscack optimzation problem to find the best lineup for a given day

Here is an example output:

------------------------- OpenOpt 0.5625 -------------------------
problem: unnamed   type: MILP    goal: max
solver: glpk
  iter  objFunVal  log10(maxResidual)  
    0  0.000e+00               0.95 
    1  0.000e+00            -100.00 
istop: 1000 (optimal)
Solver:   Time Elapsed = 0.01   CPU Time Elapsed = 0.02
objFuncValue: 137.90233 (feasible, MaxResidual = 0)
                name   Tm  Batting Order Position  Salary      preds
0   Justin Verlander  HOU            NaN        P   12200  45.444444
47         Jake Cave  MIN            8.0       OF    2500  14.558706
12         Juan Soto  WAS            2.0       OF    3700  13.247502
35        Ryan Braun  MIL            4.0       OF    3200  12.473727
11       Trea Turner  WAS            1.0       SS    3800  12.468755
75     Ronald Guzman  TEX            9.0       1B    2200  11.010237
14     Daniel Murphy  WAS            6.0       1B    2900  10.785758
40      Hernan Perez  MIL            5.0       3B    2000   9.791214
37   Jonathan Villar  MIL            1.0       2B    2500   8.121985
Predicted Points: 137.9023292329576
Salary Used: 35000