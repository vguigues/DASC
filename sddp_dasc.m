
function [lower_bounds,upper_bounds,time]=sddp_dasc(T,sampleXsi,M,n,tol,probabilities,lambda,x0,talpha,nb_iter_max)

lower_bounds=[];
upper_bounds=[];
time=[];
alphas=cell(1,T-1);
betas=cell(1,T-1);

for t=1:T-1
    alphas{1,t}=-(10)^(5);
    betas{1,t}=zeros(1,n);
end

Cum_Probas=cell(1,T);
for t=1:T
    Cum_Probas{1,t}=[0,cumsum(probabilities{1,t})];
end

End_Algo=1;
iter=1;
Costs=[];

while End_Algo
    iter
    tic
    total_cost=0;
    trial_states=cell(1,T);
    clear prob;
    for t=1:T
        if (t==1)
            Index=1;
        else
            Alea_Uniform=rand;
            [~,Index] = histc(Alea_Uniform,Cum_Probas{1,t});
            if (Alea_Uniform==1)
                Index=M;
            end
        end
                
        A=sampleXsi{1,t}(Index,1:n)'*sampleXsi{1,t}(Index,1:n)+lambda*eye(n);
        B=sampleXsi{1,t}(Index,n+1:2*n)'*sampleXsi{1,t}(Index,n+1:2*n)+lambda*eye(n);
        C=sampleXsi{1,t}(Index,n+1:2*n)'*sampleXsi{1,t}(Index,1:n);
        if (t==1)
            model.obj=[1;C*x0+sampleXsi{1,t}(Index,n+1:2*n)'];
            model.Q=sparse([0.5*[zeros(1,n+1);[zeros(n,1),B]]]);
        elseif (t==T)
            model.obj=C*trial_states{1,t-1}+sampleXsi{1,t}(Index,n+1:2*n)';
            model.Q=sparse(0.5*B);
        else
            model.obj=[1;C*trial_states{1,t-1}+sampleXsi{1,t}(Index,n+1:2*n)'];
            model.Q=sparse([0.5*[zeros(1,n+1);[zeros(n,1),B]]]);
        end
        if (t==T)
            model.rhs=[-1;1];
            model.A=sparse(2,n);
            model.A(1,:)=-ones(1,n);
            model.A(2,:)=ones(1,n);
            model.lb=zeros(n,1);
            model.ub=1*ones(n,1);
        else
            model.rhs=[-1;1;alphas{1,t}];
            model.A=sparse(2+iter,n+1);
            model.A(1,2:n+1)=-ones(1,n);
            model.A(2,2:n+1)=ones(1,n);
            for i=1:iter
                model.A(i+2,1)=1;
                model.A(i+2,2:n+1)=-betas{1,t}(i,:);
            end
            model.lb=[-Inf;zeros(n,1)];
            model.ub=[Inf;1*ones(n,1)];
        end
        model.sense='>';
        %model.modelsense=1;
        params.outputflag = 0;
        results = gurobi(model,params);
        x=results.x;
        if (t<T)
            trial_states{1,t}=x(2:n+1);
        else
            trial_states{1,t}=x;
        end
        if (t==1)
            thiscost=x(1)+0.5*x0'*A*x0+x0'*C'*trial_states{1,t}+0.5*trial_states{1,t}'*B*trial_states{1,t}+sampleXsi{1,t}(Index,1:n)*x0+sampleXsi{1,t}(Index,n+1:2*n)*x(2:n+1);
        elseif (t==T)
            thiscost=0.5*trial_states{1,t-1}'*A*trial_states{1,t-1}+trial_states{1,t}'*C*trial_states{1,t-1}+0.5*trial_states{1,t}'*B*trial_states{1,t}+sampleXsi{1,t}(Index,1:n)*trial_states{1,t-1}+sampleXsi{1,t}(Index,n+1:2*n)*x;
        else
            thiscost=x(1)+0.5*trial_states{1,t-1}'*A*trial_states{1,t-1}+trial_states{1,t}'*C*trial_states{1,t-1}+0.5*trial_states{1,t}'*B*trial_states{1,t}+sampleXsi{1,t}(Index,1:n)*trial_states{1,t-1}+sampleXsi{1,t}(Index,n+1:2*n)*x(2:n+1);
        end
        total_cost=total_cost+thiscost;
        if (t==1)
            zinf=thiscost;
            lower_bounds=[lower_bounds,zinf];
        end
    end
    
    Costs=[Costs;total_cost];
    if (iter>=200)
        zsup=mean(Costs(iter-199:iter))+talpha*sqrt(var(Costs(iter-199:iter)))/sqrt(200);
        upper_bounds=[upper_bounds,zsup];
    end
    
    %Backward pass
    
    for t=T:-1:2
        if (t==T)
            intercept=0;
            slope=zeros(n,1);
            for j=1:M
                %min 0.5*trial_states{1,t-1}'*A*trial_states{1,t-1}+x'*C*trial_states{1,t-1}+0.5*x'*B*x
                %x_t > = 0, \sum_i x_t(i)=1,
                %theta e >= \alpha_{1,t}-betas{1,t}x_t for t<T
                %Dec variable y_t=(theta,x_t) y_t=x_t for t=T
                A=sampleXsi{1,t}(j,1:n)'*sampleXsi{1,t}(j,1:n)+lambda*eye(n);
                B=sampleXsi{1,t}(j,n+1:2*n)'*sampleXsi{1,t}(j,n+1:2*n)+lambda*eye(n);
                C=sampleXsi{1,t}(j,n+1:2*n)'*sampleXsi{1,t}(j,1:n);
                model.Q=sparse(0.5*B);
                model.obj=C*trial_states{1,t-1}+sampleXsi{1,t}(j,n+1:2*n)';
                model.rhs=[-1;1];
                model.A=sparse(2,n);
                model.A(1,:)=-ones(1,n);
                model.A(2,:)=ones(1,n);
                model.lb=zeros(n,1);
                model.ub=1*ones(n,1);
                model.sense='>';
                params.outputflag = 0;
                results = gurobi(model,params);
                x=results.x;
                objective=0.5*trial_states{1,t-1}'*A*trial_states{1,t-1}+x'*C*trial_states{1,t-1}+0.5*x'*B*x+sampleXsi{1,t}(Index,1:n)*trial_states{1,t-1}+sampleXsi{1,t}(Index,n+1:2*n)*x;
                delta=A*trial_states{1,t-1}+C'*x+sampleXsi{1,t}(j,n+1:2*n)';
                objective=objective-delta'*trial_states{1,t-1};
                slope=slope+probabilities{1,t}(j)*delta;
                intercept=intercept+probabilities{1,t}(j)*objective;
            end
            alphas{1,T-1}=[alphas{1,T-1};intercept];
            betas{1,T-1}=[betas{1,T-1};slope'];
        else
            intercept=0;
            slope=zeros(n,1);
            for j=1:M
                A=sampleXsi{1,t-1}(j,1:n)'*sampleXsi{1,t-1}(j,1:n)+lambda*eye(n);
                B=sampleXsi{1,t-1}(j,n+1:2*n)'*sampleXsi{1,t-1}(j,n+1:2*n)+lambda*eye(n);
                C=sampleXsi{1,t-1}(j,n+1:2*n)'*sampleXsi{1,t-1}(j,1:n);
                model.Q=sparse(([0.5*[zeros(1,n+1);[zeros(n,1),B]]]));
                model.obj=[1;C*trial_states{1,t-1}+sampleXsi{1,t}(j,n+1:2*n)'];
                model.rhs=[-1;1;alphas{1,t}];
                model.A=sparse(3+iter,n+1);
                model.A(1,2:n+1)=-ones(1,n);
                model.A(2,2:n+1)=ones(1,n);
                for i=1:iter+1
                    model.A(2+i,1)=1;
                    model.A(2+i,2:n+1)=-betas{1,t}(i,:);
                end
                model.lb=[-(10)^50;zeros(n,1)];
                model.ub=[(10)^50;1*ones(n,1)];
                model.sense='>';
                params.outputflag = 0;
                results=gurobi(model,params);
                %t
                %j
                %iter
                %gurobi_write(model,'model.lp');
                %gurobi_iis(model,params);
                x=results.x;
                objective=x(1)+0.5*trial_states{1,t-1}'*A*trial_states{1,t-1}+x(2:n+1)'*C*trial_states{1,t-1}+0.5*x(2:n+1)'*B*x(2:n+1)+sampleXsi{1,t}(Index,1:n)*trial_states{1,t-1}+sampleXsi{1,t}(Index,n+1:2*n)*x(2:n+1);
                delta=A*trial_states{1,t-1}+C'*x(2:n+1)+sampleXsi{1,t}(j,n+1:2*n)';
                objective=objective-delta'*trial_states{1,t-1};
                slope=slope+probabilities{1,t-1}(j)*delta;
                intercept=intercept+probabilities{1,t-1}(j)*objective;
            end
            alphas{1,t-1}=[alphas{1,t-1};intercept];
            betas{1,t-1}=[betas{1,t-1};slope'];
        end
    end
    time=[time;toc];
    if (iter>=200)
    End_Algo=abs((zsup-zinf)/zsup)>tol;
    end
    if (iter>=nb_iter_max)
        End_Algo=0;
    end
    iter=iter+1;
end



