function [mu_nonlin,gamma_nonlin,mu_lin]=KFCrossValid_SVM(C,R,X,N,lambda,z_kernel,pg,pc,qc,qg_max,sg_max)


mu=[0.0001 0.001 0.01 0.1 1];
gamma_nonlin=[0.01 0.1 1 10];
T=size(z_kernel,2);

K_fold=5;
indices=crossvalind('Kfold', T, K_fold);

result=zeros(length(gamma_nonlin),length(mu));
min_eval_cost=100000;
for gamma_pos=1:length(gamma_nonlin)    
    for mu_pos=1:length(mu) 
        temp_nonlin=0;
        temp_lin=0;
        for k=1:K_fold
            
                crossvalid = (indices == k); train = ~crossvalid;
                
                z_train_kernel=z_kernel(:,train);
                z_crossvalid_kernel=z_kernel(:,crossvalid);
                pg_train=pg(:,train);
                pc_train=pc(:,train);
                qc_train=qc(:,train);
                sg_max_train=sg_max(:,train);
                qg_max_train=qg_max(:,train);
                
                pg_crossvalid=pg(:,crossvalid);
                pc_crossvalid=pc(:,crossvalid);
                qc_crossvalid=qc(:,crossvalid);
                qg_max_crossvalid=qg_max(:,crossvalid);
             
                y_train=inv(C)*(-(1-lambda)*R*qc_train+(lambda*X*R*(pg_train-pc_train))-(lambda*(X^2)*qc_train));
                y_crossvalid=inv(C)*(-(1-lambda)*R*qc_crossvalid+(lambda*X*R*(pg_crossvalid-pc_crossvalid))-(lambda*(X^2)*qc_crossvalid));
                V_RPXQc=R*(pg_train-pc_train)-X*qc_train;
                
                size_crossvalid=T/K_fold;
                size_train=T-size_crossvalid;
  
%           [a,b,qg_crossvalid,result_eachFold,eval_cost_eachFold_nonlin,K_nonlin,Obj_nonlin]=func_kernel_volReg_crossValid(C,R,X,N,lambda,mu(mu_pos),gamma_nonlin(gamma_pos),z_train_kernel,qg_max_train,z_crossvalid_kernel,qg_max_crossvalid,size_crossvalid,size_train,y_train,y_crossvalid,V_RPXQc);
%           [a,b,qg_crossvalid,result_eachFold,eval_cost_eachFold_lin,K_lin,Obj_lin]=func_kernel_volReg_crossValid(C,R,X,N,lambda,mu(mu_pos),NaN,z_train_kernel,qg_max_train,z_crossvalid_kernel,qg_max_crossvalid,size_crossvalid,size_train,y_train,y_crossvalid,V_RPXQc);  

%           [eval_cost_eachFold_nonlin]=sdpt_crossValid(C,R,X,N,lambda,mu(mu_pos),gamma_nonlin(gamma_pos),z_train_kernel,qg_max_train,z_crossvalid_kernel,qg_max_crossvalid,size_crossvalid,size_train,y_train,y_crossvalid,V_RPXQc);
%           [eval_cost_eachFold_lin]=sdpt_crossValid(C,R,X,N,lambda,mu(mu_pos),NaN,z_train_kernel,qg_max_train,z_crossvalid_kernel,qg_max_crossvalid,size_crossvalid,size_train,y_train,y_crossvalid,V_RPXQc);  
          
          [eval_cost_eachFold_nonlin]=mosek2_crossValid(C,R,X,N,lambda,mu(mu_pos),gamma_nonlin(gamma_pos),z_train_kernel,qg_max_train,z_crossvalid_kernel,qg_max_crossvalid,size_crossvalid,size_train,y_train,y_crossvalid,V_RPXQc);
          [eval_cost_eachFold_lin]=mosek2_crossValid(C,R,X,N,lambda,mu(mu_pos),NaN,z_train_kernel,qg_max_train,z_crossvalid_kernel,qg_max_crossvalid,size_crossvalid,size_train,y_train,y_crossvalid,V_RPXQc);
          
%           [eval_cost_eachFold_nonlin]=cvx2_crossValid(C,R,X,N,lambda,mu(mu_pos),gamma_nonlin(gamma_pos),z_train_kernel,qg_max_train,z_crossvalid_kernel,qg_max_crossvalid,size_crossvalid,size_train,y_train,y_crossvalid,V_RPXQc);
%           [eval_cost_eachFold_lin]=cvx2_crossValid(C,R,X,N,lambda,mu(mu_pos),NaN,z_train_kernel,qg_max_train,z_crossvalid_kernel,qg_max_crossvalid,size_crossvalid,size_train,y_train,y_crossvalid,V_RPXQc);

          temp_nonlin=temp_nonlin+eval_cost_eachFold_nonlin;
          temp_lin=temp_lin+eval_cost_eachFold_lin;
           
        end
        eval_cost_nonlin(gamma_pos,mu_pos)=temp_nonlin;
        eval_cost_lin(gamma_pos,mu_pos)=temp_lin;
    end    
end

m_nonlin=min(min(eval_cost_nonlin,[],2));
[rows_nonlin,cols_nonlin]=find(eval_cost_nonlin==m_nonlin);
m_lin=min(min(eval_cost_lin,[],2));
[rows_lin,cols_lin]=find(eval_cost_lin==m_lin);

mu_nonlin=mu(cols_nonlin);
gamma_nonlin=gamma_nonlin(rows_nonlin);

mu_lin=mu(cols_lin);

if (length(gamma_nonlin) >= 2 | length(mu_nonlin) >= 2)
    cols_nonlin=max(cols_nonlin);
    rows_nonlin=max(rows_nonlin);
    mu_nonlin=mu(cols_nonlin);
    gamma_nonlin=gamma(rows_nonlin);
end
if ( length(mu_lin) >= 2)
    cols_lin=max(cols_lin);
    rows_lin=max(rows_lin);
    mu_lin=mu(cols_lin);
end
end
