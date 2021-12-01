function[U] = make_update_palm(U,tgrad_U,L,lam,ga,fun_num)
    
    
    
    if(fun_num==1)
        U =  max(0,abs(tgrad_U) - lam/L).*sign(tgrad_U);
    end
    
    if(fun_num == 0)
        U = wthresh(tgrad_U,'h',lam/L);
        
    end
    
    
    if(fun_num==2)
        tgrad_U1 = tgrad_U(:);
        
        
        U = proximalRegC(tgrad_U1, length(tgrad_U1), lam/L, ga,1);
        
        
        U = reshape(U,size(tgrad_U,1),size(tgrad_U,2));
        
        
    end
    
    if(fun_num==3)
        tgrad_U1 = tgrad_U(:);
        
        
        U = proximalRegC(tgrad_U1, length(tgrad_U1), lam/L, ga,2);
        
        
        U = reshape(U,size(tgrad_U,1),size(tgrad_U,2));
        
    end
    
    if(fun_num==4)
        tgrad_U1 = tgrad_U(:);
        U0 = U(:);
        
        U = ProximalReg(U0, tgrad_U1, length(tgrad_U1), lam/L, ga, 0);
        
        U = reshape(U,size(tgrad_U,1),size(tgrad_U,2));
    end
    
    
    

end