function [Q,Z]=optimalGlobal(R,X,qg_max,pg,pc,qc,C,lambda,size_test,N)


Q=zeros(N,size_test);
Z=zeros(2*N,size_test);
    for i=1:size_test
        yalmip('clear');
        qt=sdpvar(N,1);
        
        
    Constraints=[];
    Constraints=[Constraints,-qg_max(:,i)<=qt];
    Constraints=[Constraints,qt<=qg_max(:,i)];

    Obj=(norm((X*(qt-qc(:,i))+R*(pg(:,i)-pc(:,i))))^2)

    ops = sdpsettings('solver','sdpt3','debug',1);
    sol_1 = optimize(Constraints,Obj,ops)
    
    qt=min(abs(qt),qg_max(:,i)).*sign(qt);
    Q(:,i)=value(qt);
    Z(:,i)=dual(Constraints);
%     res=sol_1.problem;
    end
    
end