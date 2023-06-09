classdef EuclideanParticleFilter < AbstractEuclideanFilter & AbstractParticleFilter
    % SIR particle filter on the hypertorus
    
    methods
        function this = EuclideanParticleFilter(nParticles, dim)
            % Constructor
            %
            % Parameters:
            %   nParticles (integer > 0)
            %       number of particles to use 
            %   dim (integer >0)
            %       dimension
            arguments
                nParticles (1,1) double {mustBeInteger,mustBePositive}
                dim (1,1) double {mustBeInteger,mustBePositive}
            end
            this.dist = LinearDiracDistribution(repmat((0:nParticles-1)/nParticles*2*pi,[dim,1]));
        end
        
        function setState(this, dist_)           
            assert (isa (dist_, 'AbstractLinearDistribution'));
            if ~isa(dist_, 'LinearDiracDistribution')
                dist_ = LinearDiracDistribution(dist_.sample(length(this.dist.d)));
            end
            this.dist = dist_;
        end
        
        function predictNonlinear(this, f, noiseDistribution, functionIsVectorized)
            arguments
                this (1,1) EuclideanParticleFilter
                f (1,1) function_handle
                % Can be empty for no noise, therefore do not enforce (1,1)
                noiseDistribution AbstractLinearDistribution = HypercylindricalDiracDistribution.empty
                functionIsVectorized (1,1) logical = true
            end
            predictNonlinear@AbstractParticleFilter(this, f, noiseDistribution, functionIsVectorized, false);
        end
    end
    
end

