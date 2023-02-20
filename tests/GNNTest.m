classdef GNNTest < matlab.unittest.TestCase
    properties (Constant)
        kfsInit (1,3) KalmanFilter = [KalmanFilter(GaussianDistribution(zeros(4,1), diag([1,2,3,4]))),...
            KalmanFilter(GaussianDistribution([1;2;3;4], diag([2,2,2,2]))),...
            KalmanFilter(GaussianDistribution(-[1;2;3;4], diag([4,3,2,1])))];
        measMat = [diag([1,0]), [0,0; 1,0]];
    end
    methods (Test)
        function testSetState(testCase)
            tracker = GNN();
            testCase.verifyWarningFree(@()tracker.setState(testCase.kfsInit));
        end
        function testGetState(testCase)
            tracker = GNN();
            tracker.setState(testCase.kfsInit);
            testCase.verifySize(tracker.getPointEstimate(), [4, 3]);
            testCase.verifySize(tracker.getPointEstimate(true), [12, 1]);
        end
        function testSetStateErrorOnEqual(testCase)
            % Should fail if multiple time the same
            tracker = GNN();
            allKfs = testCase.kfsInit;
            allKfs(3) = allKfs(1);
            testCase.verifyError(@()tracker.setState(allKfs), 'AbstractMultitargetTracker:MultipleCopiesInFilterBank');
        end
        function testPredictLinearAllSameMatsNoInputs(testCase)
            tracker = GNN();
            tracker.setState(testCase.kfsInit);
            tracker.predictLinear(blkdiag([1,1;0,1], [1,1;0,1]), eye(4));
            testCase.verifyEqual(tracker.filterBank(1).dist.mu, zeros(4,1));
            testCase.verifyEqual(tracker.filterBank(2).dist.mu, [3;2;7;4]);
            testCase.verifyEqual(tracker.filterBank(3).dist.mu, -[3;2;7;4]);
            testCase.verifyEqual(tracker.filterBank(1).dist.C, blkdiag([3,2; 2,2], [7,4; 4,4]) + eye(4));
            testCase.verifyEqual(tracker.filterBank(2).dist.C, blkdiag([4,2; 2,2], [4,2; 2,2]) + eye(4));
            testCase.verifyEqual(tracker.filterBank(3).dist.C, blkdiag([7,3; 3,3], [3,1; 1,1]) + eye(4));
        end
        function testPredictLinearAllSameMatsAndInputs(testCase)
            tracker = GNN();
            tracker.setState(testCase.kfsInit)
            tracker.predictLinear(blkdiag([1,1;0,1], [1,1;0,1]), eye(4), [1;-1;1;-1]);
            testCase.verifyEqual(tracker.filterBank(1).dist.mu, [1;-1;1;-1]);
            testCase.verifyEqual(tracker.filterBank(2).dist.mu, [4;1;8;3]);
            testCase.verifyEqual(tracker.filterBank(3).dist.mu, [-2;-3;-6;-5]);
            testCase.verifyEqual(tracker.filterBank(1).dist.C, blkdiag([3,2; 2,2], [7,4; 4,4]) + eye(4));
            testCase.verifyEqual(tracker.filterBank(2).dist.C, blkdiag([4,2; 2,2], [4,2; 2,2]) + eye(4));
            testCase.verifyEqual(tracker.filterBank(3).dist.C, blkdiag([7,3; 3,3], [3,1; 1,1]) + eye(4));
        end
        function testPredictLinearDifferentMatsAndInputs(testCase)
            tracker = GNN();
            tracker.setState(testCase.kfsInit);
            tracker.predictLinear(...
                cat(3, blkdiag([1,1;0,1], [1,1;0,1]), eye(4), circshift(blkdiag([1,1;0,1], [1,1;0,1]), 2)),...
                cat(3, eye(4),diag([10,11,12,13]),diag([-1,5,3,-5])),...
                cat(2, [-1;1;-1;1], [1;2;3;4], -[4;3;2;1]));
            testCase.verifyEqual(tracker.filterBank(1).dist.mu, [-1;1;-1;1]);
            testCase.verifyEqual(tracker.filterBank(2).dist.mu, [2;4;6;8]);
            testCase.verifyEqual(tracker.filterBank(3).dist.mu, [-11;-7;-5;-3]);
            testCase.verifyEqual(tracker.filterBank(1).dist.C, blkdiag([4,2; 2,3], [8,4; 4,5]));
            testCase.verifyEqual(tracker.filterBank(2).dist.C, diag([12,13,14,15]));
            testCase.verifyEqual(tracker.filterBank(3).dist.C, blkdiag([2,1; 1,6], [10,3; 3,-2]));
        end
        function testAssociationNoClutter(testCase)
            tracker = GNN();
            tracker.setState(testCase.kfsInit);
            allGaussians = [testCase.kfsInit.dist];
            % Generate perfect measurements, association should then be
            % optimal.
            perfectMeasOrdered = testCase.measMat * [allGaussians.mu];
            association = tracker.findAssociation(perfectMeasOrdered, testCase.measMat, eye(2));
            testCase.verifyEqual(association, [1,2,3]);
            % Shift them
            measurements = perfectMeasOrdered(:,[2,3,1]);
            association = tracker.findAssociation(measurements, testCase.measMat, eye(2));
            testCase.verifyEqual(measurements(:,association), perfectMeasOrdered);
            % Shift them and add a bit of noise
            measurements = perfectMeasOrdered(:,[2,3,1]) + 0.1;
            association = tracker.findAssociation(measurements, testCase.measMat, eye(2));
            testCase.verifyEqual(measurements(:,association), perfectMeasOrdered + 0.1);
            % Use different covariances
            association = tracker.findAssociation(perfectMeasOrdered(:,[2,3,1]) + 0.1, testCase.measMat,...
                cat(3, diag([1,2]), [5,0.1;0.1,3], [2,-0.5;-0.5,0.5]));
            testCase.verifyEqual(measurements(:,association), perfectMeasOrdered + 0.1);
        end
        function testAssociationWithClutter(testCase)
            tracker = GNN();
            tracker.setState(testCase.kfsInit);
            allGaussians = [testCase.kfsInit.dist];
            % Generate perfect measurements, association should then be
            % optimal.
            perfectMeasOrdered = testCase.measMat * [allGaussians.mu];
            measurements = [perfectMeasOrdered, [3;2]];
            association = tracker.findAssociation(measurements, testCase.measMat, eye(2));
            testCase.verifyEqual(association, [1,2,3]);
            % Shift them
            measurements = [perfectMeasOrdered(:,[2,3]), [2;2], perfectMeasOrdered(:,1)];
            association = tracker.findAssociation(measurements, testCase.measMat, eye(2));
            testCase.verifyEqual(measurements(:,association), perfectMeasOrdered);
            % Shift them and add a bit of noise
            measurements = [perfectMeasOrdered(:,[2,3]), [2;2], perfectMeasOrdered(:,1)] + 0.1;
            association = tracker.findAssociation(measurements, testCase.measMat, eye(2));
            testCase.verifyEqual(measurements(:,association), perfectMeasOrdered + 0.1);
            % Use different covariances
            association = tracker.findAssociation(measurements + 0.1, testCase.measMat,...
                cat(3, diag([1,2]), [5,0.1;0.1,3], [2,-0.5;-0.5,0.5], diag([2,1])));
            testCase.verifyEqual(measurements(:,association), perfectMeasOrdered + 0.1);
        end
        function testUpdateWithAndWithoutClutter(testCase)
            trackerNoClut = GNN();
            trackerClut = GNN();
            trackerNoClut.setState(testCase.kfsInit);
            trackerClut.setState(testCase.kfsInit);
            allGaussians = [testCase.kfsInit.dist];
            % Generate perfect measurements, association should then be
            % optimal.
            perfectMeasOrdered = testCase.measMat * [allGaussians.mu];
            measurementsNoClut = perfectMeasOrdered;
            trackerNoClut.updateLinear(measurementsNoClut, testCase.measMat, eye(2));
            % Mean not change because measurements are perfect
            testCase.verifyEqual([cat(2,trackerNoClut.filterBank.dist).mu],[cat(2,testCase.kfsInit.dist).mu]);
            % Covs should only decrease
            currCovs = cat(3, cat(2, trackerNoClut.filterBank.dist).C);
            testCase.verifyLessThanOrEqual(currCovs, cat(3, cat(2,testCase.kfsInit.dist).C));

            measurementsClut = [measurementsNoClut,[2;2]];
            trackerClut.updateLinear(measurementsClut, testCase.measMat, eye(2));
            testCase.verifyEqual(trackerClut.getEstimate(), trackerNoClut.getEstimate())
            
            % Shift the measurements
            measurementsNoClut = perfectMeasOrdered(:,[2,3,1]);
            trackerNoClut.updateLinear(measurementsNoClut, testCase.measMat, eye(2));
            testCase.verifyEqual([cat(3,trackerNoClut.filterBank.dist).mu],[cat(3,testCase.kfsInit.dist).mu]);
            previousCovs = currCovs;
            currCovs = cat(3, cat(2, trackerNoClut.filterBank.dist).C);
            testCase.verifyLessThanOrEqual(currCovs, previousCovs);
            
            measurementsClut = [perfectMeasOrdered(:,[2,3]), [2;2], perfectMeasOrdered(:,1)];
            trackerClut.updateLinear(measurementsClut, testCase.measMat, eye(2));
            testCase.verifyEqual(trackerClut.getEstimate(), trackerNoClut.getEstimate())
            
            % Shift the measurements and add a bit of noise
            measurementsNoClut = measurementsNoClut + 0.1;
            trackerNoClut.updateLinear(measurementsNoClut, testCase.measMat, eye(2));
            currMeans = [cat(2,trackerNoClut.filterBank.dist).mu];
            testCase.verifyNotEqual(currMeans,[cat(2,testCase.kfsInit.dist).mu]);
            previousCovs = currCovs;
            currCovs = cat(3, cat(2, trackerNoClut.filterBank.dist).C);
            testCase.verifyLessThanOrEqual(currCovs, previousCovs);
            
            measurementsClut = measurementsClut + 0.1;
            trackerClut.updateLinear(measurementsClut, testCase.measMat, eye(2));
            testCase.verifyEqual(trackerClut.getEstimate(), trackerNoClut.getEstimate());
            
            % Use different covariances
            trackerNoClut.updateLinear(measurementsNoClut, testCase.measMat,...
                cat(3, diag([1,2]), [5,0.1;0.1,3], [2,-0.5;-0.5,0.5]));
            previousMeans = currMeans;
            testCase.verifyNotEqual([cat(2,trackerNoClut.filterBank.dist).mu], previousMeans);
            previousCovs = currCovs;
            currCovs = cat(3, cat(2,trackerNoClut.filterBank.dist).C);
            for i = 1:size(currCovs,3)
                testCase.verifyLessThanOrEqual(sort(eig(currCovs(:,:,i))), sort(eig(previousCovs(:,:,i))));
            end
            
            trackerClut.updateLinear(measurementsClut, testCase.measMat,...
                cat(3, diag([1,2]), [5,0.1;0.1,3], diag([2,1]), [2,-0.5;-0.5,0.5]));
            testCase.verifyEqual(trackerClut.getEstimate(), trackerNoClut.getEstimate());
        end
        function testLogging(testCase)
            import matlab.unittest.constraints.HasNaN
            % Init
            trackerNoClut = GNN();
            trackerClut = GNN();
            trackerNoClut.setState(testCase.kfsInit);
            trackerClut.setState(testCase.kfsInit);
            testCase.verifySize(trackerNoClut.priorEstimatesOverTime, [12,1]);
            testCase.verifyEmpty(trackerNoClut.posteriorEstimatesOverTime);
            testCase.verifySize(trackerClut.priorEstimatesOverTime, [12,1]);
            testCase.verifyEmpty(trackerClut.posteriorEstimatesOverTime);
      
            % In the first time step, there should be no padding yet.
            % Hence, there should be no NaN values.
            testCase.verifyThat(trackerNoClut.priorEstimatesOverTime, ~HasNaN);
            testCase.verifyThat(trackerClut.priorEstimatesOverTime, ~HasNaN);
            % Update
            allGaussians = [testCase.kfsInit.dist];
            perfectMeasOrdered = testCase.measMat * [allGaussians.mu];
            measurementsNoClut = perfectMeasOrdered;
            measurementsClut = [perfectMeasOrdered(:,[2,3]), [2;2], perfectMeasOrdered(:,1)];
            trackerNoClut.updateLinear(measurementsNoClut, testCase.measMat, eye(2));
            trackerClut.updateLinear(measurementsClut, testCase.measMat, eye(2));

            testCase.verifySize(trackerNoClut.priorEstimatesOverTime, [12,1]);
            testCase.verifySize(trackerNoClut.posteriorEstimatesOverTime, [12,1]);
            testCase.verifySize(trackerClut.priorEstimatesOverTime, [12,1]);
            testCase.verifySize(trackerClut.posteriorEstimatesOverTime, [12,1]);
            % We know there are no NaNs
            testCase.verifyThat(trackerNoClut.posteriorEstimatesOverTime, ~HasNaN);
            testCase.verifyThat(trackerClut.posteriorEstimatesOverTime, ~HasNaN);
            % Predict
            trackerNoClut.predictLinear(blkdiag([1,1;0,1], [1,1;0,1]), eye(4), [1;-1;1;-1]);
            trackerClut.predictLinear(blkdiag([1,1;0,1], [1,1;0,1]), eye(4), [1;-1;1;-1]);

            testCase.verifySize(trackerNoClut.priorEstimatesOverTime, [12,2]);
            testCase.verifySize(trackerNoClut.posteriorEstimatesOverTime, [12,1]);
            testCase.verifySize(trackerClut.priorEstimatesOverTime, [12,2]);
            testCase.verifySize(trackerClut.posteriorEstimatesOverTime, [12,1]);
            % We know there are no NaNs
            testCase.verifyThat(trackerNoClut.priorEstimatesOverTime, ~HasNaN);
            testCase.verifyThat(trackerClut.priorEstimatesOverTime, ~HasNaN);

            allGaussians = [testCase.kfsInit.dist];
            % Generate perfect measurements, association should then be
            % optimal.
            perfectMeasOrdered = testCase.measMat * [allGaussians.mu];
            measurementsNoClut = perfectMeasOrdered;
            measurementsClut = [measurementsNoClut,[2;2]];
            trackerNoClut.updateLinear(measurementsNoClut, testCase.measMat, eye(2));
            trackerClut.updateLinear(measurementsClut, testCase.measMat, eye(2));
            testCase.verifyEqual(trackerClut.getEstimate(), trackerNoClut.getEstimate())

            testCase.verifySize(trackerNoClut.priorEstimatesOverTime, [12,2]);
            testCase.verifySize(trackerNoClut.posteriorEstimatesOverTime, [12,2]);
            testCase.verifySize(trackerClut.priorEstimatesOverTime, [12,2]);
            testCase.verifySize(trackerClut.posteriorEstimatesOverTime, [12,2]);
            testCase.verifyThat(trackerNoClut.priorEstimatesOverTime, ~HasNaN);
            testCase.verifyThat(trackerClut.priorEstimatesOverTime, ~HasNaN);
        end
    end
end
