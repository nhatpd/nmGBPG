function[U, V] = make_update_BPG(U, V, grad_U,grad_V,grad_h_U,grad_h_V,c_1,c_2,L,lam,ga,fun_num)
    
    
    if(fun_num==1)
        tpk = grad_U/L - grad_h_U;
        tqk = grad_V/L - grad_h_V;
    
        pk =  max(0,abs(tpk) - lam/L).*sign(-tpk);
        qk =  max(0,abs(tqk) - lam/L).*sign(-tqk);
        
        % solve cubic equation:
        coeff = [c_1*(norm(pk,'fro')^2 + norm(qk,'fro')^2), 0, c_2, -1];
        temp = roots(coeff);
    %     fprintf('root %.2d \n', temp);
        if(length(temp)==3)
           r = temp(3); 
        else
            r = temp;
        end

        U = r*pk;
        V = r*qk;
    end
    
    
    if(fun_num==4)
        
        
        maxiter = 30;
        
        for iter = 1:maxiter
            grad_U = grad_U - ga*lam*(1-exp(-ga*abs(U))).*sign(U);
            grad_V = grad_V - ga*lam*(1-exp(-ga*abs(V))).*sign(V);
            
            tpk = grad_U/L - grad_h_U;
            tqk = grad_V/L - grad_h_V;
            
            pk =  max(0,abs(tpk) - ga*lam/L).*sign(-tpk);
            qk =  max(0,abs(tqk) - ga*lam/L).*sign(-tqk);
            
            % solve cubic equation:
            coeff = [c_1*(norm(pk,'fro')^2 + norm(qk,'fro')^2), 0, c_2, -1];
            temp = roots(coeff);
        %     fprintf('root %.2d \n', temp);
            if(length(temp)==3)
               r = temp(3); 
            else
                r = temp;
            end

            U = r*pk;
            V = r*qk;
            
        end
    
    
    end
    
    
    

end