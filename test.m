cd 'C:\gurobi901\win64\matlab\'
gurobi_setup
addpath 'C:\Users\vince\Dropbox\Articles_Math\Vincent\Submitted\Decomp_Strongly_Convex_Value_Function';

T=20;
M=10;
lambda=100;
n=50;
sampleXsi=cell(1,T);
for t=1:T
    sampleXsi{1,t}=zeros(M,2*n);
    for i=1:M
        if (t==1)
            sampleXsi{1,t}(i,1:2*n)=0.5*ones(1,2*n);
        else
            sampleXsi{1,t}(i,1:2*n)=rand(1,2*n);
        end
    end
end

for t=1:T
    if (t==1)
        probabilities{1,t}=[1,zeros(1,M-1)];
    else
        probabilities{1,t}=(1/M)*ones(1,M);
    end
end
tol=0.1;
x0=ones(n,1);
talpha=1.96;
nb_iter_max=400;
[lower_boundss,upper_boundss,times]=sddp_dasc(T,sampleXsi,M,n,tol,probabilities,lambda,x0,talpha,nb_iter_max);
[lower_boundsd,upper_boundsd,timed]=dasc(T,sampleXsi,M,n,tol,probabilities,lambda,x0,talpha,nb_iter_max);
