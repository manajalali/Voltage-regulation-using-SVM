 function[vol,l,res]=SOCP(pgi,pci,qci,qgi,N,R,X,A,a0,r,x)

dev=.03;
v_min=1-dev;
v_min=v_min^2;
v_max=1+dev;
v_max=v_max^2;

Ap=A>0;

zs=(r.^2)+(x.^2);
% V0=sdpvar(1,1);
V0=1;
l=sdpvar(N,1,'full');
vi=sdpvar(N,1,'full');

v=V0.*a0+Ap*vi;

p=pgi-pci;
q=qgi-qci;
Pi=inv(A')*(p-diag(r)*l);
Qi=inv(A')*(q-diag(x)*l);
thetai=sdpvar(N,1,'full');
thetas = 0;
v_1=Ap*vi;
obj=(r'*l);

constraints=(A*vi+a0.*V0==(2.*diag(x)*Qi)+(2.*diag(r)*Pi)+(diag(zs)*l));

for  j=1:N
constraints=constraints+(norm([2*Pi(j);2*Qi(j);l(j)-v_1(j)-a0(j).*V0])<=l(j)+v_1(j)+a0(j).*V0);  
end
% constraints=constraints+(A*-thetai==2*(diag(r)*Qi-diag(x)*Pi)-a0*thetas);

ops=sdpsettings('solver','sedumi','verbose',0)
ops.debug=1
sol=optimize(constraints,obj,ops)
vol=value(v);
l=value(l);
value(V0);
 v1=Ap*vi;
res=sol.problem;
% Active=dual(res);
% for j=1:N
%    F(j)=norm(value(([2*Pi(j);2*Qi(j);(l(j)-v1(j)-a0(j).*V0)]))); 
%    E(j)=value(l(j)+v1(j)+a0(j).*V0);
% end
 end

 