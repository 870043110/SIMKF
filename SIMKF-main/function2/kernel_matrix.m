function omega = kernel_matrix(x, kernel_type, kerneloption, xsup)
% Construct the positive (semi-) definite and symmetric kernel matrix

[nb_data , n2]=size(x);

xsup = x;


if strcmp(kernel_type, 'RBF_kernel')
    if nargin < 4
        XXh = sum(x.^2, 2) * ones(1, nb_data);
        omega = XXh + XXh' - 2 * (x * x');
        omega = exp(-omega / (2 * kerneloption(1)));
    else
        XXh1 = sum(x.^2, 2) * ones(1, size(xsup, 1));
        XXh2 = sum(xsup.^2, 2) * ones(1, nb_data);
        omega = XXh1 + XXh2' - 2 * x * xsup';
        omega = exp(-omega / (2 * kerneloption(1)));
    end

elseif strcmp(kernel_type, 'RCG_kernel')
    [nk,nk2]=size(kerneloption);

    if nk ~=nk2
        if nk>nk2
            kerneloption=kerneloption';
        end
    else
        %  kerneloption=ones(1,n2)*kerneloption;

        % kerneloption=rand(1,n2)*n2;
        %   kerneloption=rand(1,8)*8;


    	%	kerneloption=[0.6828    0.3296    0.2070    0.5933    0.7179    0.3952];
    	% kerneloption =[17.8585   23.9033   47.4090    0.0963   40.0967   26.2703   24.7153   39.8244    8.2629   11.5658   15.9862   32.5106   22.7710 8.2558   22.8235    0.6669   24.4331    2.9067   36.5507   16.4162    4.6616   46.5630   49.9220   29.8204   38.1076    9.2414 	21.0738   19.8508   40.7777   42.1105   39.4454   17.8794   10.3472   49.9625   47.4455   36.4434   34.0924   12.8803    8.5893 	15.9071   20.7104   19.1619    2.0450    8.3309   39.6707    4.6611   28.4955   42.9951   30.2903    2.8011   28.6182];

        %   kerneloption= [ 2.5515    4.8602    6.6228    6.1235    3.4815    4.1480    4.3324    3.1411];

    end

    if length(kerneloption)~=n2 && length(kerneloption)~=n2+1
        error('Number of kerneloption is not compatible with data...');
    end


    metric = diag(1./kerneloption.^2);
    ps = x*metric*xsup';
    [nps,pps]=size(ps);
    normx = sum(x.^2*metric,2);
    normxsup = sum(xsup.^2*metric,2);
    ps = -2*ps + repmat(normx,1,pps) + repmat(normxsup',nps,1) ;
    
    omega = exp(-ps/2);
end
end




% function omega = kernel_matrix(Xtrain,kernel_type,kernel_pars,Xt)
% % Construct the positive (semi-) definite and symmetric kernel matrix
% %
% % >> Omega = kernel_matrix(X, kernel_fct, sig2)
% %
% % This matrix should be positive definite if the kernel function
% % satisfies the Mercer condition. Construct the kernel values for
% % all test data points in the rows of Xt, relative to the points of X.
% %
% % >> Omega_Xt = kernel_matrix(X, kernel_fct, sig2, Xt)
% %
% %
% % Full syntax
% %
% % >> Omega = kernel_matrix(X, kernel_fct, sig2)
% % >> Omega = kernel_matrix(X, kernel_fct, sig2, Xt)
% %
% % Outputs
% %   Omega  : N x N (N x Nt) kernel matrix
% % Inputs
% %   X      : N x d matrix with the inputs of the training data
% %   kernel : Kernel type (by default 'RBF_kernel')
% %   sig2   : Kernel parameter (bandwidth in the case of the 'RBF_kernel')
% %   Xt(*)  : Nt x d matrix with the inputs of the test data
% %
% % See also:
% %  RBF_kernel, lin_kernel, kpca, trainlssvm, kentropy
%
%
% % Copyright (c) 2011,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.be/sista/lssvmlab
%
% [nb_data,d] = size(Xtrain);
%
%
% if strcmp(kernel_type,'RBF_kernel'),
%     if nargin<4,
%         XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
%         omega = XXh+XXh'-2*(Xtrain*Xtrain');
%         omega = exp(-omega./(2*kernel_pars(1)));
%     else
%         XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
%         XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
%         omega = XXh1+XXh2' - 2*Xtrain*Xt';
%         omega = exp(-omega./(2*kernel_pars(1)));
%     end
%
% elseif strcmp(kernel_type,'RBF4_kernel'),
%     if nargin<4,
%         XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
%         omega = XXh+XXh'-2*(Xtrain*Xtrain');
%         omega = 0.5*(3-omega./kernel_pars).*exp(-omega./(2*kernel_pars(1)));
%     else
%         XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
%         XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
%         omega = XXh1+XXh2' - 2*Xtrain*Xt';
%         omega = 0.5*(3-omega./kernel_pars).*exp(-omega./(2*kernel_pars(1)));
%     end
%
% % elseif strcmp(kernel_type,'sinc_kernel'),
% %     if nargin<4,
% %         omega = sum(Xtrain,2)*ones(1,size(Xtrain,1));
% %         omega = omega - omega';
% %         omega = sinc(omega./kernel_pars(1));
% %     else
% %         XXh1 = sum(Xtrain,2)*ones(1,size(Xt,1));
% %         XXh2 = sum(Xt,2)*ones(1,nb_data);
% %         omega = XXh1-XXh2';
% %         omega = sinc(omega./kernel_pars(1));
% %     end
%
%
%
% % elseif strcmp(kernel_type,'wav_kernel')
% %     if nargin<4,
% %         XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
% %         omega = XXh+XXh'-2*(Xtrain*Xtrain');
% %
% %         XXh1 = sum(Xtrain,2)*ones(1,nb_data);
% %         omega1 = XXh1-XXh1';
% %         omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
% %
% %     else
% %         XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
% %         XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
% %         omega = XXh1+XXh2' - 2*(Xtrain*Xt');
% %
% %         XXh11 = sum(Xtrain,2)*ones(1,size(Xt,1));
% %         XXh22 = sum(Xt,2)*ones(1,nb_data);
% %         omega1 = XXh11-XXh22';
% %
% %         omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
% %     end
% end