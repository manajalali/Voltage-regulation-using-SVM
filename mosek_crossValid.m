function [eval_cost_eachFold]=mosek_crossValid(C,R,X,N,lambda,mu,gamma,z_train,qg_max_train,z_crossvalid,qg_max_crossvalid,size_crossvalid,size_train,y_train,y_crossvalid,V_RPXQc)

epsilon=0.001;%*sqrt(N);
delta=1;
clear prob;

T=size(z_train,2);

qg_max_train=qg_max_train';

G_1=zeros(N*T,N*T);
A1_i=zeros(N*T,N*T);
A3_i=zeros(N*T,T);
K=zeros(T,T,N);

    for i=1:N
        z_n=[z_train(i,:);z_train(i+N,:);z_train(i+2*N,:)];
        z_n(isnan(z_n))=0;
        z_n=[z_n;z_train(3*N+1:3*N+3,:)];
        
        reg=delta*(eye(size(z_n,2)));
        
        if (isnan(gamma))
             K(:,:,i)=z_n'*z_n+reg;
        else
            A_train=diag(exp((-diag(z_n'*z_n)/gamma)));
            B_train=exp(2*(z_n'*z_n)/gamma);
            K(:,:,i)=(A_train*B_train*A_train)+reg;
        end
    end

   for i=1:N
      G_1((i-1)*T+1:i*T,(i-1)*T+1:i*T)=K(:,:,i);
     for j=1:T
         rows=((j-1)*N+1);
         cols=((i-1)*T+1);
         A1_i(rows:rows+N-1,cols:cols+T-1)=C(:,i)*K(j,:,i);
     end  
      A3_i((i-1)*T+1:i*T,(i-1)*T+1:i*T)=sqrtm(K(:,:,i)); 
   end

G_1=[G_1 kron(eye(N),ones(T,1))];
G_1=[G_1 zeros((N*T),(N+T)) zeros(N*T,N*T) zeros(N*T,T) zeros(N*T,N*T)];
A_5=[G_1 zeros((N*T),(N+T)) zeros(N*T,N*T) zeros(N*T,T) zeros(N*T,N*T)];
h_1=vec(qg_max_train);

%Cqt+yt=y1:
Y1=kron(eye(T),eye(N));
A1_i=[A1_i kron(ones(T,1),C) zeros(T*N,N+T) -1*Y1 zeros(T*N,N*T+T)];
b1_i=-vec(y_train);

A2_i=[zeros(T,N*(T+1)) eye(T) zeros(T,N) zeros(T,N*T) -eye(T) zeros(T,N*T)];%et+epsilon=y2
b2_i=-ones(T,1)*epsilon;

Y3=kron(eye(N),eye(T));
A3_i=[A3_i zeros(N*T,2*N+T*N+2*T) -Y3];%sqrtm(K)a=y3
b3_i=zeros(N*T,1);

ind_y1=T*N+T+2*N;
ind_y2=ind_y1+N*T;
ind_y3=ind_y2+T;
ind_gamma=T*N+T+N;

SUB_1=zeros(1,(N+1)*T);
SUB_2=zeros(1,(T+1)*N);
PTR_1=zeros(1,T);
PTR_2=zeros(1,N);

for i=1:T
SUB_1((N+1)*(i-1)+1:(N+1)*i)=[ind_y2+i ind_y1+N*(i-1)+1:ind_y1+N*i];%||Cq+y||
PTR_1(i)=(N+1)*(i-1)+1;
end

for i=1:N
SUB_2((T+1)*(i-1)+1:(T+1)*i)=[ind_gamma+i ind_y3+T*(i-1)+1:ind_y3+T*i];
PTR_2(i)=(T+1)*(i-1)+1;
end

SUB=[SUB_1 SUB_2];
PTR=[PTR_1 T*(N+1)+PTR_2];

[r, res] = mosekopt('symbcon');

prob.c   = [zeros(1,(T*N)) zeros(1,N) (1/T)*ones(1,T) mu*ones(1,N) zeros(1,N*T) zeros(1,T) zeros(1,N*T)];
prob.a   = sparse([A1_i;A2_i;A3_i;G_1]);
prob.blc = [b1_i;b2_i;b3_i;-h_1];
prob.buc = [b1_i;b2_i;b3_i;h_1];
prob.blx = [-inf*ones(1,(T*N+N)) zeros(1,T) -inf*ones(1,T+2*N*T+N)]';
prob.bux = inf*ones(1,2*(N+T)+3*N*T)';

prob.cones.type   = repmat(res.symbcon.MSK_CT_QUAD,1,N+T);
prob.cones.sub    = SUB;
prob.cones.subptr = PTR;

 param.MSK_IPAR_CACHE_LICENSE = 'MSK_OFF';
[r,res]=mosekopt('minimize info',prob)

a=res.sol.itr.xx(1:N*T);
b=res.sol.itr.xx(N*T+1:N*T+N);
a=reshape(a,T,N);
a=value(a);
b=value(b);

K_crossvalid=zeros(size_crossvalid,size_train,N);
T=size(z_crossvalid,2);
qg_crossvalid=zeros(T,N);
for i=1:N
    z_n_crossvalid=[z_crossvalid(i,:);z_crossvalid(i+N,:);z_crossvalid(i+2*N,:)];
    z_n_crossvalid(isnan(z_n_crossvalid))=0;
    z_n_crossvalid=[z_n_crossvalid;z_crossvalid(3*N+1:3*N+3,:)];
    
    z_n_train=[z_train(i,:);z_train(i+N,:);z_train(i+2*N,:)];
    z_n_train(isnan(z_n_train))=0;
    z_n_train=[z_n_train;z_train(3*N+1:3*N+3,:)];
    a_i=value(a(:,i));
    a_i(isnan(a_i))=0;
    
    if (isnan(gamma))
        K_crossvalid(:,:,i)=z_n_crossvalid'*z_n_train;
        qg_crossvalid(:,i)=value(K_crossvalid(:,:,i)*a_i+b(i)*ones(T,1));
    else
        A_crossvalid=diag(exp((-diag(z_n_crossvalid'*z_n_crossvalid)/gamma)));
        B_crossvalid=exp(2*(z_n_crossvalid'*z_n_train)/gamma);
        C_crossvalid=diag(exp((-diag(z_n_train'*z_n_train)/gamma)));
        K_crossvalid(:,:,i)=A_crossvalid*B_crossvalid*C_crossvalid;

        qg_crossvalid(:,i)=value(K_crossvalid(:,:,i)*a_i+b(i)*ones(T,1));
    end
    qg_crossvalid(:,i)=min(abs(qg_crossvalid(:,i)),qg_max_crossvalid(i,:)').*sign(qg_crossvalid(:,i));
end

eval_cost_eachFold=((norm(C*qg_crossvalid'+y_crossvalid,'fro')^2)/size_crossvalid);
eval_cost_eachFold=value(eval_cost_eachFold);

end