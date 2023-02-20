classdef (Abstract) AbstractNearestNeighborTracker < AbstractMultitargetTracker
properties
    associationParam (1,1) struct
    filterBank KalmanFilter = KalmanFilter.empty
end
methods (Abstract)
    association = findAssociation(this, measurements, measurementMatrix, covMatsMeas)
end
methods
    function this = AbstractNearestNeighborTracker(initialPrior, logEstimates, associationParam)
        arguments
            initialPrior {mustBeA(initialPrior, {'KalmanFilter', 'GaussianDistribution'})} = KalmanFilter.empty();
            logEstimates (1,:) logical = true % Can set posterior and prior separately
            associationParam struct = struct()
        end
        assert(numel(logEstimates)<=2)
        if ~isempty(initialPrior)
            this.setState(initialPrior);
        end
        [this.logPosteriorEstimates, this.logPriorEstimates] = deal(logEstimates);
        if ~isempty(associationParam)
            this.associationParam = associationParam;
        end
    end
    function number = getNumberOfTargets(this)
        arguments (Input)
            this (1,1) AbstractMultitargetTracker
        end
        arguments (Output)
            number (1,1) {mustBeInteger, mustBePositive}
        end
        number = numel(this.filterBank);
    end
    function d = dim(this)
        arguments (Input)
            this (1,1) AbstractMultitargetTracker
        end
        arguments (Output)
            d (1,1) {mustBeInteger, mustBePositive}
        end
        assert(~isempty(this.filterBank), 'Cannot provide state dimension if there are no targets.')
        d = this.filterBank(1).dim;
    end
    function setState(this, newState)
        arguments
            this (1,1) AbstractMultitargetTracker
            % Can set state using either set a filter or a distribution
            % (not all may be supported for all trackers). Overload
            % function to be more restrictive
            newState {mustBeA(newState,{'AbstractFilter','AbstractDistribution'})}
        end
        if isa(newState, 'AbstractFilter')
            % Filters are stored by reference. Multiple filters should
            % not be the same object.
        assert(~any((repmat(newState, [numel(newState), 1]) == repmat(newState', [1, numel(newState)])) - eye(numel(newState)), 'all'), ...
            'AbstractMultitargetTracker:MultipleCopiesInFilterBank','No two filters of the filter bank should have the same handle. Updating the state of one target would update it for all!');
        this.filterBank = newState.copy(); % Copy to prevent manipulating original newState object
        else
            this.filterBank = arrayfun(@(dist)KalmanFilter(dist), newState);
        end
        if this.logPriorEstimates
            this.storePriorEstimates()
        end
    end
    function dists = getEstimate(this)
        arguments (Input)
            this (1,1) AbstractMultitargetTracker
        end
        arguments (Output)
            dists AbstractDistribution
        end
        if this.getNumberOfTargets()==0
            warning('Currently, there are zero targets.')
            dists = []; % When filterBank is empty, it may be [], further we cannot use .getPointEstimate() on it, so return double
        else
            dists = repmat(this.filterBank(1).getEstimate(), [1,this.getNumberOfTargets()]);
            for i=2:this.getNumberOfTargets()
                dists(i) = this.filterBank(i).getEstimate();
            end
        end
    end
    function predictLinear(this, systemMatrices, sysNoises, inputs)
        arguments
            this (1,1) AbstractNearestNeighborTracker
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
            if isa(sysNoises, 'GaussianDistribution')
                assert(all([sysNoises.mu]==0, 'all'));
                sysNoises = cat(3,sysNoises.C);
            end
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
    function updateLinear(this, measurements, measurementMatrix, covMatsMeas)
        arguments
            this (1,1) GNN
            measurements (:,:) double {mustBeNonempty}
            measurementMatrix (:,:) {mustBeNonempty} % Currently, only a single matrix is allowed
            covMatsMeas {mustBeNonempty} % Can be one for all or different for all measurements
        end
        if isempty(this.filterBank)
            warning('Currently, there are zero targets.')
            % Do nothing
            return
        end
        assert(ismatrix(measurementMatrix) && ...
            ...% Measurement matrix must be a dimMeas x dimState matrix
            isequal(size(measurementMatrix, [1,2]),...
            [size(measurements,1), size(this.filterBank(1).dist.mu,1)]),...
            'Dimensions of measurement matrix must match state and measurement dimensions.')
        association = this.findAssociation(measurements, measurementMatrix, covMatsMeas);
        
        currMeasCov = covMatsMeas; % Overwrite if multiple
        for i = 1:this.getNumberOfTargets()
            if association(i)<=size(measurements,2) % Can update only if assigned to actual measurement
                if ~ismatrix(covMatsMeas)
                    currMeasCov = covMatsMeas(:,:,association(i));
                end
                this.filterBank(i).updateLinear(measurements(:,association(i)), measurementMatrix, currMeasCov)
            end
        end
        if this.logPosteriorEstimates
            this.storePosteriorEstimates()
        end
    end
end
end