classdef KalmanFilterTest < matlab.unittest.TestCase
    
    methods(Test)
        function testInitialization(testCase)
            filterDefault = KalmanFilter();
            filter = KalmanFilter(GaussianDistribution(0,10000));
            testCase.verifyEqual(filterDefault, filter);
        end
        function testUpdateWithLikelihood1D(testCase)
            filter = KalmanFilter(GaussianDistribution(0, 1));
            filter.updateIdentity(GaussianDistribution(3, 1));
            testCase.verifyEqual(filter.getEstimate(), GaussianDistribution(1.5, 0.5));
            testCase.verifyEqual(filter.getPointEstimate(), 1.5);
        end
        function testUpdateWithMeasNoiseAndMeas1D(testCase)
            filter = KalmanFilter(GaussianDistribution(0, 1));
            filter.updateIdentity(GaussianDistribution(0, 1), 4);
            testCase.verifyEqual(filter.getEstimate(), GaussianDistribution(2, 0.5));
            testCase.verifyEqual(filter.getPointEstimate(), 2);
        end
        function testUpdateLinear2D(testCase)
            filterAdd = KalmanFilter(GaussianDistribution([0;1], diag([1,2])));
            filterId = filterAdd.copy();
            gauss = GaussianDistribution([1;0], diag([2,1]));
            filterAdd.updateLinear(gauss.mu, eye(2), gauss.C);
            filterId.updateIdentity(gauss);
            testCase.verifyEqual(filterAdd.dist, filterId.dist);
        end
        function testPredictIdentity1D(testCase)
            filter = KalmanFilter(GaussianDistribution(0, 1));
            filter.predictIdentity(GaussianDistribution(0, 3));
            testCase.verifyEqual(filter.getEstimate(), GaussianDistribution(0, 4));
            testCase.verifyEqual(filter.getPointEstimate(), 0);
        end
        function testPredictLinear2D(testCase)
            filter = KalmanFilter(GaussianDistribution([0;1], diag([1,2])));
            filter.predictLinear(diag([1,2]), diag([2,1]));
            testCase.verifyEqual(filter.dist.mu, [0;2]);
            testCase.verifyEqual(filter.dist.C, diag([3,9]));
            filter.predictLinear(diag([1,2]), diag([2,1]), [2; -2]);
            testCase.verifyEqual(filter.dist.mu, [2;2]);
        end
    end
end