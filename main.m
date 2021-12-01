clear; clc;
clear; clc;
addpath('tools')
dataset = {'movielens1m'};

load(['data/',dataset{1},'.mat']);

[row, col, val] = find(data);

[m, n] = size(data);


clear user item;


val = val - mean(val);
val = val/std(val);


para.test.m = m;
para.test.n = n;

clear m n;
clear data;

theta = 5;

para.maxR = 5;
para.maxtime = 20;

para.regType = 4;
para.maxIter = 20000;
lambda = 0.3;

para.fun_num = 4;
para.reg = 'expregularization';

para.q = 5;

para.tol = 1e-9;


rng('default');
rng(3); 
idx = randperm(length(val));

traIdx = idx(1:floor(length(val)*0.7));
tstIdx = idx(ceil(length(val)*0.3): end);

clear idx;

traData = sparse(row(traIdx), col(traIdx), val(traIdx));


para.test.row  = row(tstIdx);
para.test.col  = col(tstIdx);
para.test.data = val(tstIdx);

[m, n] = size(traData);
R = randn(n, para.maxR);
para.R = R;
para.data = dataset{1};
clear m n;
U0 = powerMethod( traData, R, para.maxR, 1e-6);
[~, ~, V0] = svd(U0'*traData, 'econ');
para.U0 = U0;
para.V0 = V0;




fprintf('runing nmGBPG \n');
method = 1;
[out{method}] = nmGBPG( traData, lambda, theta, para );

fprintf('runing BPG \n');
method = 2;
[out{method}] = BPG( traData, lambda, theta, para );

 fprintf('runing PALM \n');
method = 3;

[out{method}] = PALM( traData, lambda, theta, para ); 


fprintf('runing ADCA \n');
method = 4;
[out{method}] = ADCA( traData, lambda, theta, para ); 

figure;
subplot(1, 2, 1);
hold on;

plot(out{1}.Time, log(out{1}.obj), 'r');


plot(out{2}.Time, log(out{2}.obj), 'g');

plot(out{3}.Time, log(out{3}.obj), 'b');

plot(out{4}.Time, log(out{4}.obj), 'k');
%     



legend('nmGBPG', 'BPG','PALM','ADCA');

xlabel('CPU time (s)');
ylabel('Objective value (log scale)');
title('movielens1m')

figure;
subplot(1, 2, 1);
hold on;

plot(out{1}.Time, out{1}.RMSE, 'r');


plot(out{2}.Time, out{2}.RMSE, 'g');

plot(out{3}.Time, out{3}.RMSE, 'b');

plot(out{4}.Time, out{4}.RMSE, 'k');



legend('nmGBPG', 'BPG','PALM','ADCA');

xlabel('CPU time (s)');
ylabel('RMSE');
title('movielens1m')
