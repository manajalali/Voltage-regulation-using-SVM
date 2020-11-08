function [QG]=localControl(pg,pc,qc,qg_max,lambda,alfa)
% lambda=1;
qt=qg_max;
LEQ_l=(abs(qc)<=qt);
QGLoss=LEQ_l.*qc+(~LEQ_l).*(sign(qc)).*qt;

Delta_V=qc+(pc-pg)/alfa;
LEQ_v=(abs(Delta_V)<=qt);
QDeltaV=LEQ_v.*Delta_V+(~LEQ_v).*(sign(Delta_V)).*qt;

QGi=(1-lambda)*QGLoss+lambda*QDeltaV;
LEQ_q=(abs(QGi)<=qt);

QG=LEQ_q.*QGi+(~LEQ_q).*(sign(QGi)).*qt;
end