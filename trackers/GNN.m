classdef GNN < AbstractNearestNeighborTracker
methods
    function this = GNN(initialPrior, logEstimates, associationParam)
        arguments
            initialPrior {mustBeA(initialPrior, {'KalmanFilter', 'GaussianDistribution'})} = KalmanFilter.empty();
            logEstimates (1,:) logical = true % Can set posterior and prior separately
            associationParam (1,1) struct = struct('distanceMetricPos', 'Mahalanobis',... % distanceMetricPos can be Euclidean or Mahalanobis
                'squareDist', true, 'maxNewTracks', 10, 'gatingDistanceThreshold', chi2inv(0.999, 2).^2)
        end
        this@AbstractNearestNeighborTracker(initialPrior, logEstimates, associationParam);
    end
    function association = findAssociation(this, measurements, measurementMatrix, covMatsMeas, warnOnNoMeasForTrack)
        arguments (Input)
            this (1,1) GNN
            measurements (:,:) double {mustBeNonempty}
            measurementMatrix (:,:) {mustBeNonempty} % Must be single matrix
            covMatsMeas % Referred to as Cv in publications. 2-D or 3-D tensors (if all different) are allowed
            warnOnNoMeasForTrack (1,1) logical = true
        end
        arguments (Output)
            association (1,:) double
        end
        noTargets = size(this.filterBank, 2);
        noMeas = size(measurements, 2);
    
        assert(ismatrix(covMatsMeas) || size(covMatsMeas,3)==noMeas);
        allGaussians = [this.filterBank.dist];
        allMeansPrior = [allGaussians.mu];
        allCovMatsPrior = cat(3,allGaussians.C);
        switch this.associationParam.distanceMetricPos
            case {'euclidean','Euclidean'}
                dists=pdist2(measurements', (measurementMatrix*allMeansPrior*measurementMatrix')', 'euclidean')'; % Achtung Reihenfolge!
            case {'mahalanobis','Mahalanobis'}
                dists = NaN(noTargets, noMeas);
                
                allCovMatStateEqual = isequal(repmat(allCovMatsPrior(:,:,1), [1,1,size(allCovMatsPrior,3)]), allCovMatsPrior);
                allCovMatMeasEqual = ismatrix(covMatsMeas) || isequal(repmat(covMatsMeas(:,:,1), [1,1,size(covMatsMeas,3)]), covMatsMeas);
                
                if allCovMatMeasEqual&&allCovMatStateEqual
                    % Cp (state uncertainty) and Cv (meansurement uncertainty) is identical for all
                    currCovMahalanobis = measurementMatrix*allCovMatsPrior(:,:,1)*measurementMatrix' + covMatsMeas(:,:,1);
                    dists = pdist2((measurementMatrix*allMeansPrior)', measurements', 'mahalanobis', currCovMahalanobis);
                elseif allCovMatMeasEqual
                    % All Cv (meansurement uncertainty) are identical
                    % First determine all covariances for the Mahalanobis
                    % distance
                    allMatsMahalanobis = pagemtimes(pagemtimes(measurementMatrix,allCovMatsPrior),measurementMatrix') + covMatsMeas(:,:,1);
                    for i=1:noTargets
                        dists(i, :) = pdist2((measurementMatrix*allMeansPrior(:, i))', measurements', 'mahalanobis', allMatsMahalanobis(:,:,i));
                    end
                else
                    for i=1:noTargets
                        for j=1:noMeas
                            currCovMahalanobis = measurementMatrix*allCovMatsPrior(:,:,i)*measurementMatrix' + covMatsMeas(:,:,j);
                            dists(i, j) = pdist2((measurementMatrix*allMeansPrior(:, i))', measurements(:, j)', 'mahalanobis', currCovMahalanobis);
                        end
                    end
                end
            otherwise
                error('Association scheme not recognized');
        end
    
        if this.associationParam.squareDist
            dists = dists.^2;
        end
    
        % Pad to square and add maxNewTracks rows and columns
        padTo = max(noTargets, noMeas) + this.associationParam.maxNewTracks;
        associationMatrix = this.associationParam.gatingDistanceThreshold * ones(padTo, padTo);
        associationMatrix(1:size(dists,1), 1:size(dists,2)) = dists;
    
        [~, assignmentFull] = matlab.internal.graph.perfectMatching(associationMatrix);
        association = assignmentFull(1:noTargets)';

        if warnOnNoMeasForTrack && any(association>noMeas)
            % Assigning to something that is > noMeas means that the distance
            % was larger than the cutoff distance.
            warning('GNN:NoMeasForAtLeastOneTrack', 'No measurement was within gating threshold for at least one target.')
        end
    end
end
end