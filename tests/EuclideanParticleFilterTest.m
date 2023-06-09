classdef EuclideanParticleFilterTest< matlab.unittest.TestCase    
    methods (Test)
        function testPredictUpdateCycle3D(testCase)
            rng default
            C=[0.7,0.4,0.2;0.4,0.6,0.1;0.2,0.1,1];
            mu=[5;6;7];
            prior=GaussianDistribution(mu,C);
            pf=EuclideanParticleFilter(500,3);
            pf.setState(prior)
            forcedMean=[1;2;3];
            for i=1:50
                pf.predictIdentity(GaussianDistribution([0;0;0],C));
                testCase.verifySize(pf.getEstimateMean, [3,1]);
                pf.updateIdentity(GaussianDistribution([0;0;0],0.5*C),forcedMean);
                pf.updateIdentity(GaussianDistribution([0;0;0],0.5*C),forcedMean);
                pf.updateIdentity(GaussianDistribution([0;0;0],0.5*C),forcedMean);
            end
            testCase.verifySize(pf.getPointEstimate(), [3,1]);
            testCase.verifyEqual(pf.getPointEstimate(),forcedMean,'AbsTol',0.1);
            
            n=5;
            samples = rand(3,n);
            weights = ones(1,n)/n;
            f = @(x,w) x+w;
            pf.setState(prior)
            pf.predictNonlinearNonAdditive(f, samples, weights)
            est = pf.getEstimateMean();
            testCase.verifySize(pf.getEstimateMean, [3,1]);
            testCase.verifyEqual(est, prior.mu + mean(samples,2), 'RelTol', 0.1);
        end
        
        function testPredictUpdateCycle3DForcedPartilePosNoPred(testCase)
            rng default
            C=[0.7,0.4,0.2;0.4,0.6,0.1;0.2,0.1,1];
            mu=[1;1;1]+pi/2;
            prior=GaussianDistribution(mu,C);
            pf=EuclideanParticleFilter(500,3);
            pf.setState(prior)
            forcedMean=[1;2;3];
            forceFirstParticlePos = [1.1;2;3];
            pf.dist.d(:,1)=forceFirstParticlePos;
            for i=1:50
                testCase.verifySize(pf.getEstimateMean, [3,1]);
                pf.updateIdentity(GaussianDistribution([0;0;0],0.5*C),forcedMean);
                pf.updateIdentity(GaussianDistribution([0;0;0],0.5*C),forcedMean);
                pf.updateIdentity(GaussianDistribution([0;0;0],0.5*C),forcedMean);
            end
            testCase.verifySize(pf.getEstimateMean, [3,1]);
            testCase.verifyEqual(pf.getEstimateMean,forceFirstParticlePos,'AbsTol',1e-10);
        end
    end
end

