classdef (Abstract) AbstractEuclideanFilter < AbstractFilter
    % Abstract base class for filters on R^3
    methods
        function est = getPointEstimate(this)
            arguments
                this (1,1) AbstractEuclideanFilter
            end
            est = this.getEstimate().mean();
        end
        function mean = getEstimateMean(this)
            % Just an alternative interface. Fall back to getPointEstimate,
            % which should be overwritten by inherting functions.
            arguments
                this (1,1) AbstractEuclideanFilter
            end
            mean = this.getPointEstimate();
        end
    end
end