% Initialization

clc;
clear all;


tic
str='June_123_revised.mat'
% str='Name of the data file';
load(str);
mpc=case123_new;%load the IEEE case file

[N,R,X,A,a0,r,x]=readMat(mpc);%This funvtion generates the required matrices
N = N-1;

scale = 3;%scale the injections

even = [0 1 0 1];% a binary pattern for selecting the buses with injections
rep_scale = round(N/4);

even = repmat(even,1,rep_scale);
even = even(1:N);

pg_scaled = scale*(pg_scaled(:,1:N,1))';
pg_scaled = pg_scaled.*repmat(even',1,size(pg_scaled,2));
pg_scaled(isnan(pg_scaled))=0;

pc_scaled = scale*(pc_scaled(:,1:N,1))';
pc_scaled(isnan(pc_scaled))=0;

% Normalizing the kernel inputs:
T_start=500; %start time instance
T_final=1250;%finl time instance

over_sizing_scale = 1.1;

[P_init,P_init_kernel,sg_max_init_kernel,sg_max_init,qg_max_init_kernel,qg_max_init,pc_init,pg_init,qc_init,pc_init_kernel,pg_init_kernel,qc_init_kernel]=preprocessing(scale,over_sizing_scale,pc_scaled,qc,pg_scaled,T_start,T_final,A);

%Things that should be updated:
lambda=1;%trade-off betweenvolatge regulation and power loss
win=30; %training time window
test_win=30; %testing time window

C=sqrtm((1-lambda)*R+lambda*(X^2));

qg_final_lin_mosek=[];
qg_final_gauss_mosek=[];
qg_final_lin_mosek2=[];
qg_final_gauss_mosek2=[];
qg_final_optimal=[];
qg_final_local=[];

vol_final_local=[];
vol_lin_mosek=[];
vol_gauss_mosek=[];
vol_lin_mosek2=[];
vol_gauss_mosek2=[];
vol_final_optimal=[];
vol_final_lin_mosek=[];
vol_final_gauss_mosek=[];
vol_final_lin_mosek2=[];
vol_final_gauss_mosek2=[];
vol_final_exact_optimal=[];
vol_final_exact_local=[];
%
Cost_final_lin_mosek=[];
Cost_final_lin_mosek2=[];
Cost_final_gauss_mosek=[];
Cost_final_gauss_mosek2=[];
Cost_final_optimal=[];
Cost_final_local=[];
Z_final=[];

loc=[];
pos=1;
for pos_t=1:win:T_final-T_start
    
    epsilon=0.001;
    % Selecting the time period to be processed
    pg=pg_init(1:N,pos_t:pos_t+win-1);
    pc=pc_init(1:N,pos_t:pos_t+win-1);
    qc=qc_init(1:N,pos_t:pos_t+win-1);
    qg_max=qg_max_init(1:N,pos_t:pos_t+win-1);
    sg_max=sg_max_init(1:N,pos_t:pos_t+win-1);

    pg_kernel=pg_init_kernel(1:N,pos_t:pos_t+win-1);
    pc_kernel=pc_init_kernel(1:N,pos_t:pos_t+win-1);
    qc_kernel=qc_init_kernel(1:N,pos_t:pos_t+win-1);
    qg_max_kernel=qg_max_init_kernel(1:N,pos_t:pos_t+win-1);
    sg_max_kernel=sg_max_init_kernel(1:N,pos_t:pos_t+win-1);
    P_kernel=[P_init_kernel( 1,pos_t:pos_t+win-1);P_init_kernel(16,pos_t:pos_t+win-1);P_init_kernel(51,pos_t:pos_t+win-1)];
    
    size_pg=size(pg);
    
    z_1_kernel=(pg_kernel-pc_kernel);
    z_2_kernel=qg_max_kernel;
    z_3_kernel=qc_kernel;
    z_4_kernel=P_kernel;
    
    z_1=(pg-pc);
    z_2=qg_max;
    z_3=qc;
    
    % matrix z includes all available data (kernel inputs)
    for i=1:N
        norm_z_1(i)=norm(z_1(i,:));
        norm_z_2(i)=norm(z_2(i,:));
        norm_z_3(i)=norm(z_3(i,:));
        norm_z_1_kernel(i)=norm(z_1_kernel(i,:));
        norm_z_2_kernel(i)=norm(z_2_kernel(i,:));
        norm_z_3_kernel(i)=norm(z_3_kernel(i,:));
    end
    
    % normalizing the kernel inputs:
    norm_z=[norm_z_1';norm_z_2';norm_z_3'];
    norm_z_kernel=[norm_z_1_kernel';norm_z_2_kernel';norm_z_3_kernel'];
    
    z=[pg-pc;qg_max;qc];
    
    z_kernel=[pg_kernel-pc_kernel;qg_max_kernel;qc_kernel];
    
    z_kernel= z_kernel./repmat(norm_z_kernel,1,size(z_kernel,2));
    z_kernel(isnan(z_kernel))=0;
    
    z=z./repmat(norm_z,1,size(z,2));
    z(isnan(z))=0;
    
    T=size(z_1_kernel,2);
    
%     Finding the parameters  mu and gama using cross-validation: 
    [mu_nonlin_svm,gamma_nonlin_svm,mu_lin_svm]=KFCrossValid_SVM(C,R,X,N,lambda,z_kernel,pg,pc,qc,qg_max,sg_max);
    [mu_nonlin,gamma_nonlin,mu_lin]=KFCrossValid_SVM_nonlin(C,R,X,N,lambda,z_kernel,pg,pc,qc,qg_max);
    
    yt=inv(C)*(-(1-lambda)*R*qc+(lambda*X*R*(pg-pc))-(lambda*(X^2)*qc));
    
    V_RPXQc=R*(pg-pc)-X*qc;
    
%     Finding the parameters a and b using the mosek solver directly:
    [a_lin_mosek,b_lin_mosek,et_lin]=SVM_lin_mosek(C,N,mu_lin_svm,z_kernel,qg_max,T,yt,epsilon);
    [a_nonlin_mosek,b_nonlin_mosek,et_gauss]=SVM_gauss_mosek(C,N,mu_nonlin1_20_001(pos),z_kernel,qg_max,T,yt,gamma1_20_001(pos),epsilon);
    
    [a_lin_mosek2,b_lin_mosek2,et_lin2]=SVM2_lin_mosek(C,R,X,N,lambda,mu_lin_svm(pos),z_kernel,z,qg_max,T,yt,epsilon);
    [a_nonlin_mosek2,b_nonlin_mosek2,et_nonlin2]=SVM2_gauss_mosek(C,R,X,N,lambda,mu_nonlin1_20_001(pos),z_kernel,z,qg_max,T,yt,gamma1_20_001(pos),epsilon);
    
    clearvars pg pc qc sg_max pg_kernel pc_kernel qc_kernel sg_max_kernel P P_kernel norm_z_kernel
    
    % % % % % % % % % % % % Testing :
    test_start=pos_t+win+1;
    
    pg=pg_init(1:N,test_start:test_start+test_win-1);
    pc=pc_init(1:N,test_start:test_start+test_win-1);
    qc=qc_init(1:N,test_start:test_start+test_win-1);
    qg_max=qg_max_init(1:N,test_start:test_start+test_win-1);
    sg_max=sg_max_init(1:N,test_start:test_start+test_win-1);
    
    pg_kernel=pg_init_kernel(1:N,test_start:test_start+test_win-1);
    pc_kernel=pc_init_kernel(1:N,test_start:test_start+test_win-1);
    qc_kernel=qc_init_kernel(1:N,test_start:test_start+test_win-1);
    qg_max_kernel=qg_max_init_kernel(1:N,test_start:test_start+test_win-1);
    
    z_test=[pg-pc;qg_max;qc];
    z_test_kernel=[pg_kernel-pc_kernel;qg_max_kernel;qc_kernel];
    
    z_1_kernel=(pg_kernel-pc_kernel);
    z_2_kernel=qg_max_kernel;
    z_3_kernel=qc_kernel;
    z_1=(pg-pc);
    z_2=qg_max;
    z_3=qc;
    
    for i=1:N
        norm_z_1(i)=norm(z_1(i,:));
        norm_z_2(i)=norm(z_2(i,:));
        norm_z_3(i)=norm(z_3(i,:));
        norm_z_1_kernel(i)=norm(z_1_kernel(i,:));
        norm_z_2_kernel(i)=norm(z_2_kernel(i,:));
        norm_z_3_kernel(i)=norm(z_3_kernel(i,:));
    end
    norm_z=[norm_z_1';norm_z_2';norm_z_3'];
    norm_z_kernel=[norm_z_1_kernel';norm_z_2_kernel';norm_z_3_kernel'];
    
    z_test_kernel= z_test_kernel./repmat(norm_z_kernel,1,size(z_test_kernel,2));
    z_test= z_test./repmat(norm_z,1,size(z_test,2));
    
    z_test_kernel(isnan(z_test_kernel))=0;
    z_test(isnan(z_test))=0;
    
    yv=inv(C)*(-(1-lambda)*R*qc+(lambda*X*R*(pg-pc))-(lambda*(X^2)*qc));
    
    [qg_gauss_mosek]=eval_SVM_gauss(z_test_kernel,z_kernel,qg_max,N,test_win,gamma1_20_001(pos),a_nonlin_mosek,b_nonlin_mosek);
    [qg_lin_mosek]=eval_SVM_lin(z_test_kernel,z_kernel,qg_max,N,test_win,a_lin_mosek,b_lin_mosek);
    
    [qg_gauss_mosek2]=eval_SVM_gauss(z_test_kernel,z_kernel,qg_max,N,win,gamma1_20_001(pos),a_nonlin_mosek2,b_nonlin_mosek2,T);
    [qg_lin_mosek2]=eval_SVM_lin(z_test_kernel,z_kernel,qg_max,N,win,a_lin_mosek2,b_lin_mosek2,T);
    
    yalmip clear
    [qg_optimal,Z]=optimalGlobal(R,X,qg_max,pg,pc,qc,C,lambda,test_win,N);
    alfa=mean(x./r);
    [qg_local]=localControl(pg,pc,qc,qg_max,lambda,alfa);
    vol_exact_optimal=zeros(N,test_win);
    vol_exact_local=zeros(N,test_win);
    for i=1:test_win
        yalmip('clear');
        [vol_exact_optimal(:,i),l_exact_optimal(:,i)]=SOCP(pg(:,i),pc(:,i),qc(:,i),qg_optimal(:,i),N,R,X,A,a0,r,x);
        [vol_exact_local(:,i),l_exact_optimal(:,i)]=SOCP(pg(:,i),pc(:,i),qc(:,i),qg_local(:,i),N,R,X,A,a0,r,x);
    end
    vol_final_exact_optimal=[vol_final_exact_optimal vol_exact_optimal];
    vol_final_exact_local=[vol_final_exact_local vol_exact_local];
    
    qg_final_lin_mosek=[qg_final_lin_mosek qg_lin_mosek];
    qg_final_gauss_mosek=[qg_final_gauss_mosek qg_gauss_mosek];
    
    qg_final_lin_mosek2=[qg_final_lin_mosek2 qg_lin_mosek2];
    qg_final_gauss_mosek2=[qg_final_gauss_mosek2 qg_gauss_mosek2];
    
    qg_final_local=[qg_final_local qg_local];
    qg_final_optimal=[qg_final_optimal qg_optimal];
    Z_final=[Z_final Z];
    
    for i=1:test_win
        vol_lin_mosek(:,i)=R*(pg(:,i)-pc(:,i))+X*(qg_lin_mosek(:,i)-qc(:,i));
        c_lin_mosek(i)=lambda*norm(R*(pg(:,i)-pc(:,i))+X*(qg_lin_mosek(:,i)-qc(:,i)))^2+(1-lambda)*(qg_lin_mosek(:,i)-qc(:,i))'*R*(qg_lin_mosek(:,i)-qc(:,i))+(1-lambda)*(pg(:,i)-pc(:,i))'*R*(pg(:,i)-pc(:,i));
        
        c_gauss_mosek(i)=lambda*norm(R*(pg(:,i)-pc(:,i))+X*(qg_gauss_mosek(:,i)-qc(:,i)))^2+(1-lambda)*(qg_gauss_mosek(:,i)-qc(:,i))'*R*(qg_gauss_mosek(:,i)-qc(:,i))+(1-lambda)*(pg(:,i)-pc(:,i))'*R*(pg(:,i)-pc(:,i));
        vol_gauss_mosek(:,i)=R*(pg(:,i)-pc(:,i))+X*(qg_gauss_mosek(:,i)-qc(:,i));
        
        vol_lin_mosek2(:,i)=R*(pg(:,i)-pc(:,i))+X*(qg_lin_mosek2(:,i)-qc(:,i));
        c_lin_mosek2(i)=lambda*norm(R*(pg(:,i)-pc(:,i))+X*(qg_lin_mosek2(:,i)-qc(:,i)))^2+(1-lambda)*(qg_lin_mosek2(:,i)-qc(:,i))'*R*(qg_lin_mosek2(:,i)-qc(:,i))+(1-lambda)*(pg(:,i)-pc(:,i))'*R*(pg(:,i)-pc(:,i));
        
        c_gauss_mosek2(i)=lambda*norm(R*(pg(:,i)-pc(:,i))+X*(qg_gauss_mosek2(:,i)-qc(:,i)))^2+(1-lambda)*(qg_gauss_mosek2(:,i)-qc(:,i))'*R*(qg_gauss_mosek2(:,i)-qc(:,i))+(1-lambda)*(pg(:,i)-pc(:,i))'*R*(pg(:,i)-pc(:,i));
        vol_gauss_mosek2(:,i)=R*(pg(:,i)-pc(:,i))+X*(qg_gauss_mosek2(:,i)-qc(:,i));
        
        
        c_optimal(i)=lambda*norm(R*(pg(:,i)-pc(:,i))+X*(qg_optimal(:,i)-qc(:,i)))^2+(1-lambda)*(qg_optimal(:,i)-qc(:,i))'*R*(qg_optimal(:,i)-qc(:,i))+(1-lambda)*(pg(:,i)-pc(:,i))'*R*(pg(:,i)-pc(:,i));
        vol_optimal(:,i)=R*(pg(:,i)-pc(:,i))+X*(qg_optimal(:,i)-qc(:,i));
        
        c_local(i)=lambda*norm(R*(pg(:,i)-pc(:,i))+X*(qg_local(:,i)-qc(:,i)))^2+(1-lambda)*(qg_local(:,i)-qc(:,i))'*R*(qg_local(:,i)-qc(:,i))+(1-lambda)*(pg(:,i)-pc(:,i))'*R*(pg(:,i)-pc(:,i));
        vol_local(:,i)=R*(pg(:,i)-pc(:,i))+X*(qg_local(:,i)-qc(:,i));
    end
 
    Cost_lin_mosek=mean(c_lin_mosek);
    Cost_final_lin_mosek=[Cost_final_lin_mosek Cost_lin_mosek];
    vol_final_lin_mosek=[vol_final_lin_mosek vol_lin_mosek];
    
    Cost_gauss_mosek=mean(c_gauss_mosek);
    Cost_final_gauss_mosek=[Cost_final_gauss_mosek Cost_gauss_mosek];
    vol_final_gauss_mosek=[vol_final_gauss_mosek vol_gauss_mosek];
    
    Cost_lin_mosek2=mean(c_lin_mosek2);
    Cost_final_lin_mosek2=[Cost_final_lin_mosek2 Cost_lin_mosek2];
    vol_final_lin_mosek2=[vol_final_lin_mosek2 vol_lin_mosek2];
    
    Cost_gauss_mosek2=mean(c_gauss_mosek2);
    Cost_final_gauss_mosek2=[Cost_final_gauss_mosek2 Cost_gauss_mosek2];
    vol_final_gauss_mosek2=[vol_final_gauss_mosek2 vol_gauss_mosek2];
    
    vol_final_optimal=[vol_final_optimal vol_optimal];
    vol_final_local=[vol_final_local vol_local];
    
    Cost_optimal=mean(c_optimal);
    Cost_final_optimal=[Cost_final_optimal Cost_optimal];
    
    Cost_local=mean(c_local);
    Cost_final_local=[Cost_final_local Cost_local];
    
    loc=[loc pos_t];
    pos=pos+1;
end
toc