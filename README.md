# Voltage-regulation-using-SVM
This code is the implementation of the following paper:
M. Jalali, V. Kekatos, N. Gatsis and D. Deka,  "Designing Reactive Power Control Rules for Smart Inverters Using Support Vector Machines,"  in IEEE Transactions on Smart Grid, vol. 11, no. 2, pp. 1759-1770, March 2020, doi: 10.1109/TSG.2019.2942850. 

The main code  
1) loads the data. However, the data is not included here. the code should be modified accordingly. 
2) Finds the hyperparameters uding cross validation. While the main file includes the optimization using l2 loss function, the functions for l1 oprimization are included here with suffix "2".  
3) Solves the volatge regulation optimization problem using the Mosek solver. 
4) Solves the volatge regulation problem using the optimal power flow problem and the local control rules as well.  

The main function includes the following functions: 
1) Preprocessing: the scaling, oversizing, centering and normalzing of the data. 
2) KFCrossvalid_SVM: finds the hyperparameters using crossvalidation 
3) mosek_crossValid (mosek2_crossValid): located inside the KFCrossvalid_SVM which solves theoptimization problem 
4) SVM_gauss_mosek and SVM_lin_mosek: solve the ctual optimization problems for finding the parameters a abd b for reative power control rules.
5) localControl: finds the reactive power local control rules 
6) eval_SVM_gauss, eval_SVM_lin: evaluates the reactive power control rules given the measurements and obtained parameters 
7) optimalGlobal (SOCP): solves the central optimal power flow problem
