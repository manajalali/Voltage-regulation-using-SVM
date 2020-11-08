ufunction [a,b,et]=SVM_gauss_mosek(C,N,mu_nonlin,z_kernel,qg_max,T,yt,gamma,epsilon)

% epsilon=0.001;%*sqrt(N);
delta=1;
clear prob;

qg_max=qg_max';

G_1=zeros(N*T,N*T);
A1_i=zeros(N*T,N*T);
A3_i=zeros(N*T,T);
K=zeros(T,T,N);

 for i=1:N
    z_n=[z_kernel(i,:);z_kernel(i+N,:);z_kernel(i+2*N,:)];
    z_n(isnan(z_n))=0;
%     z_n=[z_n;z_kernel(3*N+1:3*N+3,:)];
      
    reg=delta*(eye(size(z_n,2)));
    A_train=diag(exp((-diag(z_n'*z_n)/gamma)));
    B_train=exp(2*(z_n'*z_n)/gamma);
    K(:,:,i)=(A_train*B_train*A_train)+reg;
      
    G_1((i-1)*T+1:i*T,(i-1)*T+1:i*T)=K(:,:,i);
      
    for j=1:T
      rows=((j-1)*N+1);
      cols=((i-1)*T+1);
      A1_i(rows:rows+N-1,cols:cols+T-1)=C(:,i)*K(j,:,i);
    end  
    A3_i((i-1)*T+1:i*T,(i-1)*T+1:i*T)=sqrtm(K(:,:,i));
 end

G_1=[G_1 kron(eye(N),ones(T,1))];
% G_1=[G_1 zeros((N*T),(N+T)) zeros(N*T,N*T) zeros(N*T,T) zeros(N*T,N*T)];
G_1=[G_1 zeros((N*T),(N+T)) zeros(N*T,N*T) zeros(N*T,T) zeros(N*T,N*T)]; %This is for a=[a b]
h_1=vec(qg_max);

%Cqt+yt=y1:
Y1=kron(eye(T),eye(N));

A1_i=[A1_i kron(ones(T,1),C) zeros(T*N,N+T) -1*Y1 zeros(T*N,N*T+T)];
% A1_i=[A1_i zeros(T*N,N+T) -1*Y1 zeros(T*N,N*T+T)];
b1_i=-vec(yt);

A2_i=[zeros(T,N*(T+1)) eye(T) zeros(T,N) zeros(T,N*T) -eye(T) zeros(T,N*T)];%et+epsilon=y2
% A2_i=[zeros(T,N*(T)) eye(T) zeros(T,N) zeros(T,N*T) -eye(T) zeros(T,N*T)];%et+epsilon=y2
b2_i=ones(T,1)*(-epsilon);

Y3=kron(eye(N),eye(T));

A3_i=[A3_i zeros(N*T,2*N+T*N+2*T) -Y3];%sqrtm(K)a=y3
% A3_i=[A3_i zeros(N*T,N+T*N+2*T) -Y3];%sqrtm(K)a=y3
b3_i=zeros(N*T,1);

ind_y1=T*N+T+2*N;
% ind_y1=T*N+T+N;
ind_y2=(2*T*N)+T+(2*N);
% ind_y2=(2*T*N)+T+(N);
ind_y3=ind_y2+T;
ind_gamma=T*N+T+N;
% ind_gamma=T*N+T;

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

[r,res]=mosekopt('symbcon');

prob.c   = [zeros(1,(T*N)) zeros(1,N) (1/T)*ones(1,T) mu_nonlin*ones(1,N) zeros(1,N*T) zeros(1,T) zeros(1,N*T)];
% prob.c   = [zeros(1,(T*N)) (1/T)*ones(1,T) mu_nonlin*ones(1,N) zeros(1,N*T) zeros(1,T) zeros(1,N*T)];
prob.a   = sparse([A1_i;A2_i;A3_i;G_1]);
prob.blc = [b1_i;b2_i;b3_i;-h_1];
prob.buc = [b1_i;b2_i;b3_i;h_1];
prob.blx = [-inf*ones(1,(T*N+N)) zeros(1,T+N) -inf*ones(1,T+2*N*T)]';
prob.bux = [inf*ones(1,(T*N+N)) inf*ones(1,T) inf*ones(1,T+2*N*T+N)]';
% prob.blx = [-inf*ones(1,(T*N)) zeros(1,T+N) -inf*ones(1,T+2*N*T)]';
% prob.bux = [inf*ones(1,(T*N)) inf*ones(1,T) inf*ones(1,T+2*N*T+N)]';

prob.cones.type   = repmat(res.symbcon.MSK_CT_QUAD,1,N+T);
prob.cones.sub    = SUB;
prob.cones.subptr = PTR;

%  param.MSK_IPAR_CACHE_LICENSE = 'MSK_OFF';
[r,res]=mosekopt('minimize',prob)

a=res.sol.itr.xx(1:N*T);
b=res.sol.itr.xx(N*T+1:N*T+N);
% b=zeros(N,1);
et=res.sol.itr.xx(N*T+N+1:N*T+N+T);
a=reshape(a,T,N);
a=value(a);
b=value(b);
et=value(et);

end