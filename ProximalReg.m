function [w] = ProximalReg(w_0, v, d, lam, theta, type)
%minimize 1/2|x-v|^2 + lam * sum_i(1-exp(-theta*|x_i|)
if type==0
    maxiter = 100;
    fun = zeros(maxiter+1,1);
    fun(1) = ((norm(w_0-v))^2)/2 + lam*sum(1-exp(-theta*abs(w_0)));
    w = w_0;
    for iter = 1:maxiter
        grad = v + lam*theta*(1-exp(-theta*abs(w))).*sign(w);
        %w = SoftThreshold(grad,lam*theta);
        w = max(0,abs(grad) - lam*theta).*sign(grad);
        fun(iter+1) = ((norm(w-v))^2)/2 + lam*sum(1-exp(-theta*abs(w)));
        relativediff = abs(fun(iter) - fun(iter+1))/fun(iter+1);
        %fprintf('objective: %.3f %0.6f %0.10f \n',iter, fun(iter+1),fun(iter) - fun(iter+1));
         if relativediff < 1e-5 && iter > 5
             break
         end
    end
else
    w = proximalRegC(v, d, lam, theta,type);
end