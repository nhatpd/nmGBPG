function[ObjVal] = main_func(data, part, U, V, lam, fun_num, ga)

    if(fun_num==0)
        ObjVal = (1/2)*sum((data - part').^2) + lam*(nnz(U) + nnz(V));

    end
    
    if(fun_num==5)
        if(nnz(U) <= lam && nnz(V) <=lam)
            
            ObjVal = (1/2)*sum((data - part').^2) + lam*(nnz(U) + nnz(V));
        else
            ObjVal = inf;
        end
        

    end
    
    if(fun_num == 1)
        ObjVal = (1/2)*sum((data - part').^2) +  lam*(sum(sum(abs(U))) + sum(sum(abs(V))));
       
    end
    
    if(fun_num == 2)
        U = U(:);
        V = V(:);
        ObjVal = (1/2)*sum((data - part').^2) + funRegC(U,length(U),lam,ga,1) + funRegC(V,length(V),lam,ga,1);
       
    end
    
    if(fun_num == 3)
        U = U(:);
        V = V(:);
        ObjVal = (1/2)*sum((data - part').^2) + funRegC(U,length(U),lam,ga,2) + funRegC(V,length(V),lam,ga,2);
       
    end
    
    if(fun_num == 4)
        ObjVal = (1/2)*sum((data - part').^2) + lam*(sum(sum(1 - exp(-ga*abs(U)))) + sum(sum(1 - exp(-ga*abs(V)))));
    end

end