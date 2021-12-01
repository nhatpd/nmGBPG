function [output ] = BPG( D, lam, theta, para )
% D: sparse observed matrix

% work well
output.method = 'BPG';

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(D));
end

objstep = 1;

maxIter = para.maxIter;
tol = para.tol*objstep;

regType = para.regType;

[row, col, data] = find(D);

[m, n] = size(D);

% U = randn(size(D, 1), 1);
% V0 = randn(size(D, 2), 1);
% V1 = V0;
% S = 1;

%R = randn(n, maxR);
%U0 = powerMethod( D, R, 5, 1e-6);
R = para.R;
U0 = para.U0;
U1 = U0;

% [~, ~, V0] = svd(U0'*D, 'econ');
% V0 = V0';
V0 = para.V0';
V1 = V0;

spa = sparse(row, col, data, m, n); % data input == D

c_1 = 3;
c_2 = norm(D,'fro');

clear D;

obj = zeros(maxIter+1, 1);
RMSE = zeros(maxIter+1, 1);
trainRMSE = zeros(maxIter+1, 1);
Time = zeros(maxIter+1, 1);
nnzUV = zeros(maxIter+1, 2);

part1 = partXY(U0', V0, row, col, length(data));

ga = theta;
fun_num = para.fun_num;

obj(1) = main_func(data, part1, U0, V0, lam, fun_num, ga);


L = 1; %0.1
if(isfield(para, 'test'))
    tempS = eye(size(U1,2), size(V1',2));
    if(para.test.m ~= m)
        RMSE(1) = MatCompRMSE(V1', U1, tempS, para.test.row, para.test.col, para.test.data);
        trainRMSE(1) = sqrt(sum((data - part1').^2)/length(data));
    else
        RMSE(1) = MatCompRMSE(U1, V1', tempS, para.test.row, para.test.col, para.test.data);
        trainRMSE(1) = sqrt(sum((data - part1').^2)/length(data));
    end
    fprintf('method: %s data: %s  RMSE %.2d \n', output.method, para.data, RMSE(1));
end


for i = 1:maxIter
    tt = cputime;
    %spa is the gradient of objective
    
    
    
    % compute gradient
    
    part0 = data - part1';
    setSval(spa,part0,length(part0));
    
%     fprintf('size 1 U: %d; size 2 U: %d ; size 1 spa %d; size spa 2: %d\n', ...
%         size(y_V,1), size(y_V,2), size(spa,1), size(spa,2));

    grad_U = -spa*V1';
    grad_V = -U1'*spa;
    
%     if(fun_num==4)
%         grad_U = grad_U - ga*lam*(1-exp(-ga*abs(U1))).*sign(U1);
%         grad_V = grad_V - ga*lam*(1-exp(-ga*abs(V1))).*sign(V1);
%     end
    
    norm_y = norm(U1,'fro')^2 + norm(V1,'fro')^2;
    grad_h1_U = U1*norm_y;
    grad_h1_V = V1*norm_y;

    grad_h2_U = U1;
    grad_h2_V = V1;

    grad_h_U = c_1*grad_h1_U + c_2*grad_h2_U;
    grad_h_V = c_1*grad_h1_V + c_2*grad_h2_V;
    
      
        % update U, V 
    [U1, V1] = make_update_BPG(U1,V1,grad_U,grad_V,grad_h_U,grad_h_V,c_1,c_2,L,lam, ga, fun_num);



    part1 = sparse_inp(U1', V1, row, col);

    x_obj = main_func(data, part1, U1, V1, lam, fun_num, ga);
        

    if(i > 1)
        delta = (obj(i)- x_obj)/x_obj;
    else
        delta = inf;
    end
    
    Time(i+1) = cputime - tt;
    obj(i+1) = x_obj;
    
    fprintf('iter: %d; obj: %.3d (dif: %.3d); rank %d; lambda: %.1f; L %d; time: %.3d;  nnz U:%0.3d; nnz V %0.3d \n', ...
        i, x_obj, delta, para.maxR, lam, L, Time(i+1), nnz(U1)/(size(U1,1)*size(U1,2)),nnz(V1)/(size(V1,1)*size(V1,2)));
    
    nnzUV(i+1,1) = nnz(U1)/(size(U1,1)*size(U1,2));
    nnzUV(i+1,2) = nnz(V1)/(size(V1,1)*size(V1,2));
    
    
    
    % testing performance
    if(isfield(para, 'test'))
        tempS = eye(size(U1,2), size(V1',2));
        if(para.test.m ~= m)
            RMSE(i+1) = MatCompRMSE(V1', U1, tempS, para.test.row, para.test.col, para.test.data);
            trainRMSE(i+1) = sqrt(sum((data - part1').^2)/length(data));
        else
            RMSE(i+1) = MatCompRMSE(U1, V1', tempS, para.test.row, para.test.col, para.test.data);
            trainRMSE(i+1) = sqrt(sum((data - part1').^2)/length(data));
        end
        fprintf('method: %s data: %s  RMSE %.2d \n', output.method, para.data, RMSE(i));
    end
    
%     if(i > 1 && abs(delta) < tol)
%         break;
%     end
    
    if(sum(Time) > para.maxtime)
        break;
    end
end

output.obj = obj(1:(i+1));
% [U0, S, V] = svd(U1, 'econ');
% V = V1*V;
output.Rank = para.maxR;
output.RMSE = RMSE(1:(i+1));
output.trainRMSE = trainRMSE(1:(i+1));

Time = cumsum(Time);
output.Time = Time(1:(i+1));
output.U = U1;
output.V = V1;
output.data = para.data;
output.L = L;
% output.Ils = Ils(1:(i+1));
output.nnzUV = nnzUV(1:(i+1),:);
output.lambda = lam;
output.theta = ga;
output.reg = para.reg;

end


