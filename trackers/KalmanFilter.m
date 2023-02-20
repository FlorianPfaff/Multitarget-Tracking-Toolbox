classdef KalmanFilter < AbstractEuclideanFilter
    properties
        dist (1,1) GaussianDistribution = GaussianDistribution(0,10000) % Default value
    end
    
    methods
        function this = KalmanFilter(prior)
            arguments
                prior (1,1) GaussianDistribution = GaussianDistribution(0,10000);
            end
            this.dist = prior;
        end
        
        function setState(this, dist)
            arguments
                this (1,1) KalmanFilter
                dist (1,1) GaussianDistribution
            end
            assert(isa(dist, 'GaussianDistribution'));
            this.vm = dist;
        end
        
        function predictIdentity(this, sysNoise)
            arguments (Input)
                this (1,1) KalmanFilter
                sysNoise (1,1) GaussianDistribution
            end
            this.dist = this.dist.convolve(sysNoise);
        end

        function predictLinear(this, systemMatrix, sysNoise, input)
            arguments
                this (1,1) KalmanFilter
                systemMatrix (:,:) double {mustBeNonempty}
                sysNoise {mustBeA(sysNoise,{'GaussianDistribution','double'}), mustBeNonempty}
                input (:,1) double = zeros(size(this.dist.mu))
            end
            if isempty(input) % For being called by MTT with empty input
                input = zeros(size(this.dist.mu));
            end
            if isa(sysNoise, 'GaussianDistribution')
                % If given as Gaussian, extract relevant parameters
                if any(input~=0)
                    warning('Noise has nonzero mean, check if this is intended. It is now added to the input.')
                    input = input + sysNoise.mu;
                end
                sysNoise = sysNoise.C;
            end
            muNew = systemMatrix*this.dist.mu + input;
            Cnew = systemMatrix*this.dist.C*systemMatrix' + sysNoise;
            % modify to prevent using constructor
            this.dist.mu = muNew;
            this.dist.C = Cnew;
        end
        
        function updateIdentity(this, distMeas, z)
            arguments (Input)
                this (1,1) KalmanFilter
                distMeas (1,1) GaussianDistribution
                z (:,1) double = []
            end
            if ~isempty(z)
                distMeas = distMeas.shift(z);
            end
            this.dist = this.dist.multiply(distMeas);
        end

        function updateLinear(this, measurement, measurementMatrix, covMatMeas)
            arguments (Input)
                this (1,1) KalmanFilter
                measurement (:,1) double
                measurementMatrix (:,:) double
                covMatMeas (:,:) double
            end
            KalmanGain = this.dist.C*measurementMatrix'/(covMatMeas+measurementMatrix*this.dist.C*measurementMatrix');
            x = this.dist.mu + KalmanGain*(measurement-measurementMatrix*this.dist.mu);
            C = this.dist.C-KalmanGain*measurementMatrix*this.dist.C;

            this.dist.mu = x;
            this.dist.C = C;
        end
        
        function dist = getEstimate(this)
            arguments(Input)
                this (1,1) KalmanFilter
            end
            arguments (Output)
                dist (1,1) GaussianDistribution
            end
            dist = this.dist;
        end
    end
    
end

