% This funtion is used to solve a class of problems like \min_\Phi
% \|G-\Psi'\Phi'\Phi\Psi\|_F^2 +lambda \|\Phi E\|_F^2 where G belongs to an equiangular tight frame and
% \lambda is the trade-off parameter.
% E is the sparse representation error and can be
% omitted.
% See the following paper:
% Tao Hong and Zhihui Zhu,  
% ``An Efficient Method for Robust Projection Matrix Design''.
%=======================================
%Input:
%        param: a struct to define all of the necessary parameters in this
%        function.
%        param.M: the number of rows of the sensing matrix. : $M$
%        param.Psi: the dictionary: $M \times N$
%        param.L_iter: the number of iterations regarding the alternating.
%        The default is 200.
%        param.mu: the valuable of the shrinkage, 
%                  if it is omitted, the welch-bound is given, i.e.
%                  sqrt((L-M)/M/(L-1)). 
%        param.paramCG: the parameter in CG method. 
%                       We need to install the software minFunc for activating CG.
%        param.isfigure: If it is true, three figures will be plotted in
%                        time regarding the mu, mav and objective value versus
%                        iteration. The default is false.
%        param.lambda: the trade-off parameter. The default is 0.1;
%        param.E: If E exists, we utilize the model which has sparse
%        representation error.
%Output:
%           Phi_n: the sensing matrix after optimization
%           evolution_mx: the evolution of maximun mu versus the iteration.
%           evolution_mav: the evolution of the average mu versus the
%           iteration.
%           F_obj: The objective value of the given problem versus every
%           alternating iteration.
%==========================================================================
%Example:
% [Phi_n,evolution_mx,evolution_mav,F_obj] = Projection_matrix_ETF(param);
%
% Last modified version is given by T. Hong, in Taub Technion, 26th, Apr.
% 2018.
% @Copyright.
%==========================================================================
function [Phi_n,evolution_mx,evolution_mav,F_obj] = Projection_matrix_ETF(param)

epsilon = 1e-6;
if isfield(param,'Psi')
    Psi = param.Psi;
else
    error('The dictionary is not given');
end

if isfield(param,'M')
    M = param.M;
else
    error('The dimension of projection matrix is not given');
end

[N,L] = size(Psi);

if abs(norm(Psi(:,randperm(L,1)))-1)>epsilon
    fprintf('The column of dictioanry need to be normalized\n');
    Psi=Psi*diag(1./sqrt(sum(Psi.*Psi)));
end

if isfield(param,'L_iter')
    L_iter = param.L_iter;
else
    L_iter = 200;
end

if isfield(param,'mu')
    mu = param.mu;
else
    mu = sqrt((L-M)/M/(L-1));
end

if isfield(param,'paramCG')
    paramCG = param.paramCG;
else
    paramCG = struct('Method','cg','display','off','LS',0,'TolX',1e-20,'TolFun',1e-20,'MaxIter',5e3,'optTol',1e-20);%'TolX',1e-12,'TolFun',1e-12;
end

if isfield(param,'isfigure')
    isfigure = param.isfigure;
else
    isfigure = false;
end

if isfield(param,'E')
    E = param.E;
    isE = true;
else
    isE = false;
end

evolution_mx = [];
evolution_mav = [];
F_obj = [];

if isfield(param,'lambda')
    lambda = param.lambda;
else
    lambda = 0.1;
end

rng(2,'v5uniform')
Phi = randn(M,N);  
%wait = waitbar(0,'Proposed alternating iteration is beginning...');
if isfigure
    figure
end

for l = 1:L_iter
    % waitbar(l/L_iter,wait)
    D_eq = Phi*Psi;
    G = D_eq'*D_eq;
    
    D_eq_tilde = D_eq*diag(1./sqrt(sum(D_eq.*D_eq)));
    G_tilde = D_eq_tilde'*D_eq_tilde;
    G_tilde_off_d = abs(G_tilde-diag(diag(G_tilde))); G_tilde_off_d = G_tilde_off_d(:); index = G_tilde_off_d>mu;
    evolution_mx(end+1) = max(G_tilde_off_d);
    evolution_mav(end+1) = sum(G_tilde_off_d(index))/sum(index); % This equation is used to calculate the average of \mu the defination is given in Elad's paper and it does not use the defination in AFS's paper.
    
    G = G-diag(diag(G));
    index = abs(G)>mu;
    G(index) = sign(G(index))*mu;
    G = G+eye(L);
    G_t = G;
    if isE
        Phi = minFunc(@(x)Projection_Matrix_BFGS_E(x,Psi,lambda,G_t,M,E),Phi(:),paramCG);
        Phi = reshape(Phi,M,N);
        F_obj = [F_obj;4*Projection_Matrix_BFGS_E(Phi,Psi,lambda,G_t,M,E)];
    else
        Phi = minFunc(@(x)Projection_Matrix_BFGS(x,Psi,lambda,G_t,M),Phi(:),paramCG);
        F_obj = [F_obj;4*Projection_Matrix_BFGS(Phi,Psi,lambda,G_t,M)];
    end
    if isfigure
        subplot(221)
        plot(1:length(evolution_mav),evolution_mav)
        title('The average mutual coherence')
        xlabel('Iteration');
        z = ylabel('$\mu_{av}$');
        set(z,'interpreter','latex');
        drawnow;
        subplot(222)
        plot(1:length(evolution_mx),evolution_mx)
        title('The maximal mutual coherence')
        xlabel('Iteration');
        z = ylabel('$\mu_{mx}$');
        set(z,'interpreter','latex');
        drawnow;
        subplot(223)
        semilogy(1:length(F_obj),F_obj)
        xlabel('Iteration');
        z = ylabel('$f(\Phi_k)$');
        set(z,'interpreter','latex');
        drawnow;
    end
end
%close(wait)
Phi_n = Phi;

return

function [f_obj,grad] = Projection_Matrix_BFGS(Phi_v,D,lambda,G,M)
[N,~] = size(D);
Phi = reshape(Phi_v,M,N);
f_obj = norm(G-D'*(Phi'*Phi)*D,'fro')^2+lambda*norm(Phi,'fro')^2;
f_obj = f_obj/4;
if nargout>1
    grad = -Phi*(D*G*D')+Phi*(D*D')*(Phi'*Phi)*(D*D')+(0.5*lambda)*Phi;
    grad = grad(:);
end
return

function [f_obj,grad] = Projection_Matrix_BFGS_E(Phi_v,D,lambda,G,M,E)
[N,~] = size(D);
Phi = reshape(Phi_v,M,N);
f_obj = norm(G-D'*(Phi'*Phi)*D,'fro')^2+lambda*norm(Phi*E,'fro')^2;
f_obj = f_obj/4;
if nargout>1
    grad = -Phi*(D*G*D')+Phi*(D*D')*(Phi'*Phi)*(D*D')+(0.5*lambda)*Phi*(E*E');
    grad = grad(:);
end
return
