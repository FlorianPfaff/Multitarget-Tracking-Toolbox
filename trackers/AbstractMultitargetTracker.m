classdef (Abstract) AbstractMultitargetTracker < AbstractFilter
properties
    logPriorEstimates (1,1) logical = true
    logPosteriorEstimates (1,1) logical = true
    priorEstimatesOverTime (:,:) double
    posteriorEstimatesOverTime (:,:) double
end
methods
    function storePriorEstimates(this)
        arguments (Input)
            this (1,1) AbstractMultitargetTracker
        end
        currEsts = this.getPointEstimate(true);
        if size(currEsts, 1)<=size(this.priorEstimatesOverTime, 1)
            % Add (padded) estimate to matrix
            this.priorEstimatesOverTime = [this.priorEstimatesOverTime, [currEsts,NaN(size(this.priorEstimatesOverTime, 1) - size(currEsts, 1), 1)]];
        else % Current estimate is longer. Create lager matrix first
            priorEstimatesOverTimeNew = NaN(size(currEsts, 1), size(this.priorEstimatesOverTime,2)+1);
            priorEstimatesOverTimeNew(1:size(this.priorEstimatesOverTime,1), 1:size(this.priorEstimatesOverTime,2));
            priorEstimatesOverTimeNew(:, end) = currEsts;
            this.priorEstimatesOverTime = priorEstimatesOverTimeNew;
        end
    end
    function storePosteriorEstimates(this)
        arguments (Input)
            this (1,1) AbstractMultitargetTracker
        end
        currEst = this.getPointEstimate(true);
        if size(currEst, 1)<=size(this.posteriorEstimatesOverTime, 1)
            % Add (padded) estimate to matrix
            this.posteriorEstimatesOverTime = [this.posteriorEstimatesOverTime, [currEst,NaN(size(this.posteriorEstimatesOverTime, 1) - size(currEst, 1), 1)]];
        else % Current estimate is longer. Create lager matrix first
            posteriorEstimatesOverTimeNew = NaN(size(currEst, 1), size(this.posteriorEstimatesOverTime,2)+1);
            posteriorEstimatesOverTimeNew(1:size(this.posteriorEstimatesOverTime,1), 1:size(this.posteriorEstimatesOverTime,2))...
                = this.posteriorEstimatesOverTime;
            posteriorEstimatesOverTimeNew(:, end) = currEst;
            this.posteriorEstimatesOverTime = posteriorEstimatesOverTimeNew;
        end
    end

    function pointEsts = getPointEstimate(this, flattenVector)
        arguments (Input)
            this (1,1) AbstractMultitargetTracker
            flattenVector (1,1) logical = false
        end
        arguments (Output)
            pointEsts double
        end
        if this.getNumberOfTargets()==0
            warning('Currently, there are zero targets.')
            pointEsts = []; % Cannot return NaN(dim, 0) because We have no idea about the dimension when filterBank is empty
        else
            pointEsts = NaN(this.dim, this.getNumberOfTargets());
            for i=1:this.getNumberOfTargets()
                pointEsts(:, i) = this.filterBank(i).getPointEstimate();
            end
            if flattenVector
                pointEsts = pointEsts(:);
            end
        end
    end
    function predictLinear(this, systemMatrices, sysNoises, inputs)
        arguments
            this (1,1) AbstractMultitargetTracker
            systemMatrices double {mustBeNonempty}
            sysNoises {mustBeA(sysNoises,{'GaussianDistribution','double'}),mustBeNonempty}
            inputs double = []
        end
        if isempty(this.filterBank)
            warning('Currently, there are zero targets.')
            % Do nothing
        else
            assert(all(size(systemMatrices,[1,2])==this.filterBank(1).dim) && ndims(systemMatrices)<=3,...
                'May be a single (dimSingleState, dimSingleState) matrix or a (dimSingleState, dimSingleState, noTargets) tensor.')
            % Set to inputs for now. If there are multiple, overwrite in
            % loop
            currSysMatrix = systemMatrices;
            currSysNoise = sysNoises;
            currInput = inputs;
            for i = 1:this.getNumberOfTargets()
                if ~ismatrix(systemMatrices)
                    currSysMatrix = systemMatrices(:,:,i);
                end
                if ~ismatrix(sysNoises)
                    currSysNoise = sysNoises(:,:,i);
                end
                if size(inputs,2) > 1
                    currInput = inputs(:,i);
                end
                this.filterBank(i).predictLinear(currSysMatrix, currSysNoise, currInput)
            end
            if this.logPriorEstimates
                this.storePriorEstimates()
            end
        end
    end
end
end

