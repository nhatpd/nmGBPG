function [output ] = ADCA( D, lam, theta, para )
% D: sparse observed matrix

% work well
output.method = 'ADCA';

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
c_2 = norm(data);

clear D;

obj = zeros(maxIter+1, 1);
obj_y = zeros(maxIter+1, 1);
RMSE = zeros(maxIter+1, 1);
trainRMSE = zeros(maxIter+1, 1);
Time = zeros(maxIter+1, 1);
Lls = zeros(maxIter+1, 1);
Ils = zeros(maxIter+1, 1);
nnzUV = zeros(maxIter+1, 2);
no_acceleration = zeros(maxIter+1, 1);

part0 = partXY(U0', V0, row, col, length(data));
% part1 = partXY(U1', V1, row, col, length(data));

ga = theta;
fun_num = para.fun_num;
% fun_num = 1;
obj(1) = main_func(data, part0, U0, V0, lam, fun_num, ga);
obj_y(1) = obj(1);
c = 1;

L = 1;
sigma = 1e-5;

maxinneriter = 300;

Lls(1) = L;
 % testing performance
if(isfield(para, 'test'))
    tempS = eye(size(U1,2), size(V1',2));
    if(para.test.m ~= m)
        RMSE(1) = MatCompRMSE(V1', U1, tempS, para.test.row, para.test.col, para.test.data);
        trainRMSE(1) = sqrt(sum((data - part0').^2)/length(data));
    else
        RMSE(1) = MatCompRMSE(U1, V1', tempS, para.test.row, para.test.col, para.test.data);
        trainRMSE(1) = sqrt(sum((data - part0').^2)/length(data));
    end
    fprintf('method: %s data: %s  RMSE %.2d RMSE %.2d \n', output.method, para.data, RMSE(1),obj(1));
end

% part1 = data - part0';
% setSval(spa,part1,length(part0));
% 
% gradL_U = -spa*V1';
% gradL_V = -U1'*spa;



% t_min = 1e-20;
% t_max = 1e20;
cp = 1;
c = cp;

for i = 1:maxIter
    tt = cputime;
%     c = 1;
    %spa is the gradient of objective
    bi = (c - 1)/(c + 2);
%     c = (1+sqrt(4*cp^2+1))/2;
%     bi = (cp - 1)/c;

    % compute Y
    y_U = (1+bi)*U1 - bi*U0;
    y_V = (1+bi)*V1 - bi*V0;
    
    part1 = sparse_inp(y_U', y_V, row, col);
    
    y_obj = main_func(data, part0, y_U, y_V, lam, fun_num, ga);
    
    if(y_obj > getMaxOverk(obj(1:i), para.q))
        y_U = U1;
        y_V = V1;
        part1 = part0;
        y_obj = obj(i);
        no_acceleration(i) = i;
    end
    obj_y(i) = y_obj;
    U0 = U1;
    V0 = V1;
    
    
    % compute gradient
    
    part0 = data - part1';
    setSval(spa,part0,length(part0));
    
%     fprintf('size 1 U: %d; size 2 U: %d ; size 1 spa %d; size spa 2: %d\n', ...
%         size(y_V,1), size(y_V,2), size(spa,1), size(spa,2));


    
    
% --------------------------

    grad_U = -spa*y_V';
    grad_V = -y_U'*spa;
    
    if(fun_num==4)
        grad_U = grad_U - lam*ga*(1-exp(-ga*abs(y_U))).*sign(y_U);
        grad_V = grad_V - lam*ga*(1-exp(-ga*abs(y_V))).*sign(y_V);
    end
    
    
% --------------------------
    
    norm_y = norm(y_U,'fro')^2 + norm(y_V,'fro')^2;
    grad_h1_U = y_U*norm_y;
    grad_h1_V = y_V*norm_y;

    grad_h2_U = y_U;
    grad_h2_V = y_V;

    grad_h_U = c_1*grad_h1_U + c_2*grad_h2_U;
    grad_h_V = c_1*grad_h1_V + c_2*grad_h2_V;
    
%     obj_h_y = c_1*0.25*(norm_y^2) + c_2*0.5*norm_y;
    
    [U1, V1] = make_update(grad_U,grad_V,grad_h_U,grad_h_V,c_1,c_2,L,lam,ga, fun_num);
        
%     norm_x = norm(U1,'fro')^2 + norm(V1,'fro')^2;

%     delta_U = U1 - y_U;
%     delta_V = V1 - y_V;
% 
%     obj_h_x = c_1*0.25*(norm_x^2) + c_2*0.5*norm_x;

%     reg = obj_h_x - obj_h_y - sum(sum(delta_U.*grad_h_U)) - sum(sum(delta_V.*grad_h_V));

    part0 = sparse_inp(U1', V1, row, col);

    x_obj = main_func(data, part0, U1, V1, lam, fun_num, ga);
    
    Lls(i+1) = L;
%     Ils(i+1) = inneriter;
    
    %initialize L
%     if inneriter == 1
%         L = L*0.9;
%     end
%     gradL_U_prev = gradL_U;
%     gradL_V_prev = gradL_V;
%     
%     part1 = data - part0';
%     setSval(spa,part1,length(part1));
%     
%     gradL_U = -spa*V1';
%     gradL_V = -U1'*spa;
%     
%     
%     st = norm(U1 - U0,'fro')^2 + norm(V1 - V0,'fro')^2;
%     stg = sum(sum((U1 - U0).*(gradL_U - gradL_U_prev))) + sum(sum((V1 - V0).*(gradL_V - gradL_V_prev)));
%     if(st>1e-12)
%         L = min(t_max,max(t_min,stg/st));
%     end
    
    % ----------------------
    c = c + 1;
%     cp = c;
    

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
            trainRMSE(i+1) = sqrt(sum((data - part0').^2)/length(data));
        else
            RMSE(i+1) = MatCompRMSE(U1, V1', tempS, para.test.row, para.test.col, para.test.data);
            trainRMSE(i+1) = sqrt(sum((data - part0').^2)/length(data));
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
output.L = Lls(1:(i+1));
% output.Ils = Ils(1:(i+1));
output.nnzUV = nnzUV(1:(i+1),:);
output.no_acceleration = no_acceleration(1:(i+1));
output.lambda = lam;
output.theta = ga;
output.reg = para.reg;


end


function[pk] = HS(U,lam)

    tpk = U(:);
    [~,ind] = sort(tpk,'ascend');
    tpk(ind(1:(end-lam))) = 0;
    pk = reshape(tpk,size(U,1),size(U,2));

end

