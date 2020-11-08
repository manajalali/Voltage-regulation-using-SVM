function [P_init,P_init_kernel,sg_max_init_kernel,sg_max_init,qg_max_init_kernel,qg_max_init,pc_init,pg_init,qc_init,pc_init_kernel,pg_init_kernel,qc_init_kernel]=preprocessing(scale,over_sizing_scale,pc_scaled,qc,pg_scaled,start,final,A)

P_scaled = (inv(A))'*(pg_scaled-pc_scaled);
P_var    = sqrt(var(P_scaled')');
P_mean   = mean(P_scaled')';
P_kernel = (P_scaled-repmat(P_mean,1,size(P_scaled,2)))./(repmat(P_var,1,size(P_scaled,2)));
P_kernel(isnan(P_kernel))=0;

pc_var=sqrt(var(pc_scaled')');
pc_mean=mean(pc_scaled')';
pc_kernel=(pc_scaled-repmat(pc_mean,1,size(pc_scaled,2)))./(repmat(pc_var,1,size(pc_scaled,2)));
pc_kernel(isnan(pc_kernel))=0;

pg_var=sqrt((var(pg_scaled'))');
pg_mean=mean(pg_scaled')';
pg_kernel=(pg_scaled-repmat(pg_mean,1,size(pg_scaled,2)))./(repmat(pg_var,1,size(pg_scaled,2)));
pg_kernel(isnan(pg_kernel))=0;

qc_scaled=scale*(qc(:,:,1))';
qc_scaled(isnan(qc_scaled))=0;

qc_var=sqrt(var(qc_scaled')');
qc_mean=mean(qc_scaled')';

qc_kernel=(qc_scaled-repmat(qc_mean,1,size(qc,1)))./(repmat(qc_var,1,size(qc,1)));
qc_kernel(isnan(qc_kernel))=0;

pg_init=pg_scaled(:,start:final);
pg_init_kernel=(pg_kernel(:,start:final));
pg_max=max(pg_scaled')';
pg_max_kernel=max(pg_kernel')';

P_init=P_scaled(:,start:final);
P_init_kernel=P_kernel(:,start:final);
pc_init=pc_scaled(:,start:final);
pc_init_kernel=pc_kernel(:,start:final);
qc_init=qc_scaled(:,start:final);
qc_init_kernel=qc_kernel(:,start:final);

sg_max_init=over_sizing_scale*pg_max;
sg_max_init=repmat(sg_max_init,1,size(pg_init,2));
qg_max_init=(realsqrt((sg_max_init.^2)-(pg_init.^2)));
sg_max_init_kernel=1.1*pg_max_kernel;
sg_max_init_kernel=repmat(sg_max_init_kernel,1,size(pg_init_kernel,2));
qg_max_init_kernel=(realsqrt(sg_max_init_kernel.^2-(pg_init_kernel.^2)));
    
end