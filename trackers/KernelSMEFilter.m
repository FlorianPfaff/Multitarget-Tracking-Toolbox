classdef KernelSMEFilter < AbstractMultitargetTracker
properties
    x (:,1) double
    C (:,:) double
    nTargets (1,1) {mustBeNonnegative} = 0
end
methods
    function this = KernelSMEFilter(initialPriors, logEstimates)
        arguments
            initialPriors (1,:) GaussianDistribution = GaussianDistribution.empty()
            logEstimates (1,:) logical = true % Can set posterior and prior separately
        end
        if ~isempty(initialPriors)
            this.setState(initialPriors);
        end
        [this.logPosteriorEstimates, this.logPriorEstimates] = deal(logEstimates);
    end
    function est = getEstimate(this)
        arguments (Input)
            this (1,1) KernelSMEFilter
        end
        est = GaussianDistribution(this.x, this.C);
    end
    function est = getPointEstimate(this, flatVector)
        arguments (Input)
            this (1,1) KernelSMEFilter
            flatVector (1,1) logical = false
        end
        arguments (Output)
            est double
        end
        est = this.x;
        if ~flatVector
            est = reshape(est, [], this.nTargets);
        end
    end

    function this = setState(this, priors)
        arguments
            this (1,1) KernelSMEFilter
            priors (1,:) GaussianDistribution {mustBeNonempty}
        end
        this.nTargets = numel(priors);
        xCell = arrayfun(@(prior){prior.mu}, priors);
        CCell = arrayfun(@(prior){prior.C}, priors);
        this.x = cat(1,xCell{:});
        this.C = blkdiag(CCell{:});
        if this.logPriorEstimates
            this.storePriorEstimates()
        end
    end
    function number = getNumberOfTargets(this)
        arguments (Input)
            this (1,1) AbstractMultitargetTracker
        end
        arguments (Output)
            number (1,1) {mustBeInteger, mustBePositive}
        end
        number = this.nTargets;
    end
    function predictLinear(this, systemMatrix, sysNoise, inputs)
        arguments
            this (1,1) KernelSMEFilter
            % Same motion model must be used for all targets in the Kernel SME.
            systemMatrix (:,:) double {mustBeNonempty}
            sysNoise {mustBeA(sysNoise,{'GaussianDistribution','double'}),mustBeNonempty}
            inputs double = []
        end
        assert(isempty(inputs), 'Inputs are currently not supported for the Kernel SME filter');
        if isa(sysNoise, 'GaussianDistribution')
            sysNoiseCov = sysNoise.C;
        else
            sysNoiseCov = sysNoise;
        end
        this.x = reshape(systemMatrix*reshape(this.x, [], this.nTargets), [], 1);
        sysMatCell = repmat({systemMatrix}, 1, this.nTargets);
        sysNoiseCovCell = repmat({sysNoiseCov}, 1, this.nTargets);
        this.C = blkdiag(sysMatCell{:})*this.C*blkdiag(sysMatCell{:})' + blkdiag(sysNoiseCovCell{:});
        if this.logPriorEstimates
            this.storePriorEstimates();
        end
    end
    function d = dim(this)
        arguments (Output)
            d (1,1) double {mustBeInteger, mustBePositive}
        end
        d = size(this.x,1)/this.nTargets;
    end
    function updateLinear(this, measurements, measurementMatrix, covMatMeas, falseAlarmRate, clutterCov, lambdaMultimeas, enableGating, gatingThreshold)
        arguments
            this (1,1) KernelSMEFilter
            measurements (:,:) double {mustBeNonempty, mustBeNonNan}
            measurementMatrix (:,:) double {mustBeNonempty, mustBeNonNan}
            covMatMeas (:,:) double {mustBeNonempty} % Expect measurement noise to be the same for all measurements
            falseAlarmRate (1,1) = 0 % Set to zero to disable clutter
            clutterCov (:,:) = zeros(size(measurements,1))
            lambdaMultimeas = 1 % Set to 1 when no multimeasurements (formulae for lambdaMultimeas = 1 are identical to those without multimeasurements)
            enableGating (1,1) logical = false
            gatingThreshold (1,1) double = chi2inv(0.99, size(measurements,1))
        end
        assert(enableGating||gatingThreshold==chi2inv(0.99, size(measurements,1)),...
            'Changed gating threshold without enabling gating, this does not make sense.');
        nMeas = size(measurements, 2);    
        kernelwidth = mean(diag(covMatMeas))^2;

        if enableGating
            gti = @(target)((target-1)*this.dim+1):(target*this.dim);
            dists = NaN(this.nTargets, nMeas);
            for i = 1:this.nTargets
                dists(i, :) = pdist2((measurementMatrix * this.x(gti(i)))', measurements',...
                    'mahalanobis', measurementMatrix*this.C(gti(i),gti(i))*measurementMatrix' + covMatMeas);
            end
            dists = dists.^2; % Squared Mahalanobis distances are usually considered
            % Filter out all that are incompatible with all tracks
            measurements = measurements(:, any(dists<gatingThreshold));
        end

        testPoints = KernelSMEFilter.genTestPoints(measurements, kernelwidth);
        pseudoMeas = KernelSMEFilter.calcPseudoMeas(testPoints, measurements, kernelwidth);

        [mu_s, sigma_s, sigma_xs] = KernelSMEFilter.calcMoments(... 
            this.x, this.C, measurementMatrix, covMatMeas, testPoints, ...
            kernelwidth, this.nTargets, falseAlarmRate, clutterCov, lambdaMultimeas);

        if cond(sigma_s) > 1e+10
            sigma_s = sigma_s + 1e-5 * eye(size(sigma_s, 1));
        end

        xPosterior = this.x + sigma_xs * (sigma_s\(pseudoMeas-mu_s));
        this.C = this.C - sigma_xs * (sigma_s\sigma_xs');
        this.x = xPosterior;

        if this.logPosteriorEstimates
            this.storePosteriorEstimates()
        end
    end
end
methods (Static)
    function testPoints = genTestPoints(measurements, kernelwidth)
        arguments (Input)
            measurements (:,:) double {mustBeNonempty, mustBeNonNan}
            kernelwidth (1,1) double {mustBePositive}
        end
        arguments (Output)
            testPoints (:,:) double {mustBeNonNan}
        end
        measDim = size(measurements,1);
        nMeas = size(measurements,2);
        testPoints = NaN(size(measurements,1), (1+2*measDim)*nMeas);
        for i = 1:nMeas
            xt = measurements(:,i);
            dCell = repmat({sqrt(2*kernelwidth)*[-1,1]}, [1,measDim]);
            testPoints(:,(i-1)*(2*measDim+1)+1:i*(2*measDim+1)) = [xt, xt+blkdiag(dCell{:})];
        end
    end
    function pseudoMeas = calcPseudoMeas(testPoints, measurements, kernelWidth)
        % Function supports scenarios with clutter
        arguments (Input)
            testPoints (:,:) double {mustBeNonempty, mustBeNonNan}
            measurements (:,:) double {mustBeNonempty, mustBeNonNan}
            kernelWidth (1,1) double {mustBePositive}
        end
        arguments (Output)
            pseudoMeas (:,1) double {mustBeNonNan}
        end
        nMeas = size(measurements, 2);
        measDim = size(measurements, 1);
        nTestPoints = size(testPoints, 2);
        pseudoMeas = zeros(nTestPoints, 1);
        for i = 1:nTestPoints
            a_i = testPoints(:, i);
            for l=1:nMeas
                pseudoMeas(i) = pseudoMeas(i) + mvnpdf(a_i', measurements(:, l)', kernelWidth*eye(measDim));
            end
        end
    end
    function [mu_s, sigma_s, sigma_xs] = calcMoments(xPrior, CPrior, measurementMatrix, covMatMeas,...
            testPoints, kernelwidth, nTargets, falseAlarmRate, clutterCov, lambdaMultimeas)
        arguments (Input)
            % State should be given as a stateDim * nTargets x 1 vector
            xPrior (:,1) double {mustBeNonNan}
            % Give full covariance matrix, i.e., stateDim * nTargets x stateDim * nTargets
            CPrior (:,:) double {mustBeNonNan}
            measurementMatrix (:,:) double {mustBeNonNan}
            covMatMeas (:,:) double {mustBeNonNan}
            testPoints (:,:) double {mustBeNonNan}
            kernelwidth (1,1) double {mustBePositive}
            nTargets (1,1) double {mustBePositive, mustBeInteger}
            falseAlarmRate (1, 1) double {mustBeNonnegative} = 0 % Set to zero when no clutter 
            clutterCov (:, :) double {mustBeNonNan} = zeros(size(covMatMeas))
            % Set lambdaMultimeas to 1 when no multimeasurements (formulae for lambdaMultimeas = 1 are identical to those without multimeasurements)
            % Can provide vector for different values for the individual
            % targets or use single value to use the same for all.
            lambdaMultimeas (1,:) double = 1
        end
        arguments (Output)
            mu_s (:,1) double {mustBeNonNan}
            sigma_s (:,:) double {mustBeNonNan}
            sigma_xs (:,:) double {mustBeNonNan}
        end
        if numel(lambdaMultimeas)==1
            lambdaMultimeas = lambdaMultimeas * ones(1,nTargets);
        end
        nTestpoints = size(testPoints, 2);
        assert(all(size(covMatMeas)==size(measurementMatrix,1)));
        stateDim = numel(xPrior)/nTargets;
        gti = @(target)((target-1)*stateDim+1):(target*stateDim);
        
        %%% Step 3a: Pre-calculate required mvnpdf evaluations
        P = NaN(nTestpoints, nTargets);
        for l = 1:nTargets
            sTemp = measurementMatrix * CPrior(gti(l),gti(l)) * measurementMatrix' + covMatMeas + kernelwidth * eye(size(covMatMeas));
            muTemp = (measurementMatrix * xPrior(gti(l)))';
            for i = 1:size(testPoints,2) 
                P(i,l) =  mvnpdf(testPoints(:,i)', muTemp, sTemp) + falseAlarmRate * mvnpdf(testPoints(:,i)',...
                    zeros(1, size(measurementMatrix,1)), clutterCov + kernelwidth * eye(size(measurementMatrix,1)));
            end
        end
            
        Pij = zeros(nTestpoints,nTestpoints, nTargets);
        for l = 1:nTargets
            sTemp = measurementMatrix * CPrior(gti(l),gti(l)) * measurementMatrix' + covMatMeas + 0.5*kernelwidth * eye(size(covMatMeas));
            muTemp = (measurementMatrix * xPrior(gti(l)))';
            for i = 1:nTestpoints
                for j = 1:size(testPoints,2)
                    Pij(i,j,l) =  mvnpdf(0.5*(testPoints(:,i)' + testPoints(:,j)'),  muTemp,  0.5*(sTemp + sTemp')) ;
                end
            end
        end
        
        %%% Step 3b: Calculate mean of predicted pseudo-measurement
        mu_s = sum(P,2);

        %%%% Step 3c: Calculate covariance of predicted pseudo-measurement
        sigma_s = zeros(size(testPoints,2));
        for i = 1:nTestpoints
            for j = i:nTestpoints
                M = sum(P(j,:));    
                P_temp = mvnpdf(testPoints(:,i)', testPoints(:,j)', 2 * kernelwidth * eye(size(covMatMeas)));
                
                sigma_s(i,j) = sum(lambdaMultimeas .* (P_temp * reshape(Pij(i,j,:),1,[]) + P(i,:) .* (M - P(j,:))))...
                   + sigma_s(i,j) + ...
                   falseAlarmRate^2 * mvnpdf(testPoints(:,i)', zeros(1, size(measurementMatrix,1)), clutterCov + kernelwidth * eye(size(covMatMeas)))...
                   * mvnpdf(testPoints(:,j)', zeros(1, size(measurementMatrix,1)), clutterCov + kernelwidth * eye(size(covMatMeas)))...
                   + falseAlarmRate * mvnpdf(testPoints(:,i)', testPoints(:,j)', 2 * kernelwidth * eye(size(covMatMeas)))...
                   * mvnpdf(0.5*(testPoints(:,i) + testPoints(:,j))', zeros(1, size(measurementMatrix,1)), clutterCov + kernelwidth * eye(size(covMatMeas)))...
                   - mu_s(i) * mu_s(j) - falseAlarmRate^2 ...
                   * mvnpdf(testPoints(:,i)', zeros(1, size(measurementMatrix,1)), clutterCov + kernelwidth * eye(size(covMatMeas)))...
                   * mvnpdf(testPoints(:,j)', zeros(1, size(measurementMatrix,1)), clutterCov + kernelwidth * eye(size(covMatMeas)));
            end
        end
        % Make symmetric
        sigma_s = sigma_s + triu(sigma_s, 1)';

        %%% Step 3d: Calculate cross-covariance of predicted pseudo-measurement
        sigma_xs = zeros(stateDim * nTargets, size(testPoints,2));
        
        %%% Calculate Kalman gains
        K = cell(1,nTargets);
        for  l = 1:nTargets
            K{l} = CPrior(:,gti(l)) * measurementMatrix'/...
                (measurementMatrix * CPrior(gti(l),gti(l)) * measurementMatrix' + covMatMeas + kernelwidth * eye(size(covMatMeas)));
        end
        
        for i = 1:size(testPoints,2) 
            for l = 1:nTargets
                sigma_xs(:,i) = sigma_xs(:,i) + lambdaMultimeas(l) * P(i,l) * (xPrior + K{l} * (testPoints(:,i) - measurementMatrix * xPrior(gti(l))));
            end
            sigma_xs(:,i) = sigma_xs(:,i) - mu_s(i) * xPrior;
        end
    end
end
end