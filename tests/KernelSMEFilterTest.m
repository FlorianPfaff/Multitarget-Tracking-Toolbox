classdef KernelSMEFilterTest < matlab.unittest.TestCase
    properties (Constant)
        gaussians2DCV (1,3) GaussianDistribution = [GaussianDistribution(zeros(4,1), diag(1:4)),...
            GaussianDistribution(1:4, 2*eye(4)),...
            GaussianDistribution(-(1:4), diag(4:-1:1))]
        measMat2DCV = [diag([1,0]), [0,0; 1,0]]

        gaussians3DCV (1,3) GaussianDistribution = [GaussianDistribution(zeros(6,1), diag(1:6)),...
            GaussianDistribution(1:6, 2*eye(6)),...
            GaussianDistribution(-(1:6), diag(6:-1:1))]
        measMat3DCV = [[diag([1,0]);0,0], [diag(1,-1);0,0], [0,0;diag(1,-1)]]

        gaussians3DCA (1,3) GaussianDistribution = [GaussianDistribution(zeros(9,1), diag(1:9)),...
            GaussianDistribution(1:9, 2*eye(9)),...
            GaussianDistribution(-(1:9), diag(9:-1:1))]
        measMat3DCA = [diag([1,0,0]), [0,0,0; 1,0,0; 0,0,0], diag(1,-2)]

        scenarioNames = {'2DCV', '3DCV', '3DCA'};
    end
    methods (Test)
        function testSetState(testCase)
            for i = 1:numel(testCase.scenarioNames)
                tracker = KernelSMEFilter();
                testCase.verifyWarningFree(@()tracker.setState(testCase.(['gaussians',testCase.scenarioNames{i}])));
            end
        end
        function testGetState(testCase)
            tracker = KernelSMEFilter();
            tracker.setState(testCase.gaussians2DCV);
            testCase.verifySize(tracker.getPointEstimate(), [4, 3]);
            testCase.verifySize(tracker.getPointEstimate(true), [12, 1]);
            tracker.setState(testCase.gaussians3DCV);
            testCase.verifySize(tracker.getPointEstimate(), [6, 3]);
            testCase.verifySize(tracker.getPointEstimate(true), [18, 1]);
            tracker.setState(testCase.gaussians3DCA);
            testCase.verifySize(tracker.getPointEstimate(), [9, 3]);
            testCase.verifySize(tracker.getPointEstimate(true), [27, 1]);
        end
        function testPredictLinearAllSameMatsNoInputs2DCV(testCase)
            tracker = KernelSMEFilter();
            tracker.setState(testCase.gaussians2DCV);
            tracker.predictLinear(blkdiag([1,1;0,1], [1,1;0,1]), eye(4));
            testCase.verifyEqual(tracker.x, [zeros(4,1);[3;2;7;4];-[3;2;7;4]]);
            testCase.verifyEqual(tracker.C(:,:), blkdiag(...
                [3,2; 2,2], [7,4; 4,4], [4,2; 2,2], [4,2; 2,2], [7,3; 3,3], [3,1; 1,1]) + eye(12));
        end
        function testPredictLinearAllSameMatsNoInputs3DCV(testCase)
            tracker = KernelSMEFilter();
            tracker.setState(testCase.gaussians3DCV);
            tracker.predictLinear(blkdiag([1,1;0,1], [1,1;0,1], [1,1;0,1]), eye(6));
            testCase.verifyEqual(tracker.x, [zeros(6,1);[3;2;7;4;11;6];-[3;2;7;4;11;6]]);
            testCase.verifyEqual(tracker.C(:,:), blkdiag(...
                [3,2; 2,2], [7,4; 4,4], [11,6; 6,6], [4,2; 2,2], [4,2; 2,2], [4,2; 2,2], [11,5; 5,5], [7,3; 3,3], [3,1; 1,1]) + eye(18));
        end
        
        function testUpdateIndependentOfOrder(testCase)
            for i = 1:numel(testCase.scenarioNames)
                trackerOrdered = KernelSMEFilter();
                trackerUnordered = KernelSMEFilter();
                trackerOrdered.setState(testCase.(['gaussians',testCase.scenarioNames{i}]));
                trackerUnordered.setState(testCase.(['gaussians',testCase.scenarioNames{i}]));
                allGaussians = [testCase.(['gaussians',testCase.scenarioNames{i}])];
                % Generate perfect measurements, association should then be
                % optimal.
                perfectMeasOrdered = testCase.(['measMat',testCase.scenarioNames{i}]) * [allGaussians.mu];
                measurements = perfectMeasOrdered;
                trackerOrdered.updateLinear(measurements, testCase.(['measMat',testCase.scenarioNames{i}]), eye(size(measurements,1)));
                % Mean should not change a lot because means have been measured
                testCase.verifyEqual(trackerOrdered.getPointEstimate(),[cat(2,testCase.(['gaussians',testCase.scenarioNames{i}])).mu], 'AbsTol', 0.3);
                % Covs should improve, i.e., different should be approximately
                % positive definite (add tolerance of up to 1-e10)
                covTrackerAfterFirstUpdate = trackerOrdered.C;
                testCase.verifyLessThanOrEqual(-1e-10, eig(blkdiag(cat(2,testCase.(['gaussians',testCase.scenarioNames{i}])).C) - covTrackerAfterFirstUpdate));
                % Do another update step
                trackerOrdered.updateLinear(measurements, testCase.(['measMat',testCase.scenarioNames{i}]), eye(size(measurements,1)));
                % Result should still be reasonably close to the original state
                testCase.verifyEqual(trackerOrdered.getPointEstimate(),[cat(2,testCase.(['gaussians',testCase.scenarioNames{i}])).mu], 'AbsTol', 0.4);
                % Covariance should have improved again
                testCase.verifyLessThanOrEqual(-1e-10, eig(covTrackerAfterFirstUpdate - trackerOrdered.C));
    
                trackerUnordered.updateLinear(measurements(:, [2,3,1]), testCase.(['measMat',testCase.scenarioNames{i}]), eye(size(measurements,1)));
                trackerUnordered.updateLinear(measurements(:, [1,3,2]), testCase.(['measMat',testCase.scenarioNames{i}]), eye(size(measurements,1)));
                testCase.verifyEqual(trackerOrdered.getPointEstimate(),trackerUnordered.getPointEstimate(), 'AbsTol', 0.3);
            end
        end
        
        function testUpdateWithClutter(testCase)
            % When adding a clutter measurement that is not filtered out via gating, states should be drawn
            % toward the clutter measurement
            tracker = KernelSMEFilter();
            tracker.setState(testCase.gaussians2DCV);
            perfectMeasOrdered = testCase.measMat2DCV * [testCase.gaussians2DCV.mu];

            % We pull the targets a bit toward [2;2]
            measurementsUnorderedWithClutter = [perfectMeasOrdered(:,[2,3]), [2;2], perfectMeasOrdered(:,1)];
            tracker.updateLinear(measurementsUnorderedWithClutter, testCase.measMat2DCV, eye(2));
            
            priorMeans = [testCase.gaussians2DCV.mu];
            posteriorMeansFirstUpdate = tracker.getPointEstimate();
            testCase.verifyGreaterThanOrEqual(sum(vecnorm(priorMeans([1,3],:)-[2;2])), 10);
            testCase.verifyLessThanOrEqual(sum(vecnorm(posteriorMeansFirstUpdate([1,3],:)-[2;2])), 8);

            % Now we pull the targets a toward -[2;2]
            measurementsUnorderedWithClutter = [perfectMeasOrdered(:,[2,3]), -[2;2], perfectMeasOrdered(:,1)];
            for i=1:15
                tracker.updateLinear(measurementsUnorderedWithClutter, testCase.measMat2DCV, eye(2));
            end

            posteriorMeansAfterAllUpdates = tracker.getPointEstimate();
            testCase.verifyLessThanOrEqual(sum(vecnorm(posteriorMeansAfterAllUpdates([1,3],:)+[2;2])), 8);
            testCase.verifyGreaterThanOrEqual(sum(vecnorm(posteriorMeansFirstUpdate([1,3],:)+[2;2])), 10);
            testCase.verifyGreaterThanOrEqual(sum(vecnorm(priorMeans([1,3],:)-[2;2])), 10);
        end

        function testGating(testCase)
            trackerWithGating = KernelSMEFilter();
            trackerNoGating = KernelSMEFilter();
            trackerWithGating.setState(testCase.gaussians2DCV);
            trackerNoGating.setState(testCase.gaussians2DCV);
            perfectMeasOrdered = testCase.measMat2DCV * [testCase.gaussians2DCV.mu];

            % [2;2] is within gating threshold. Hence, they should behave
            % identically.
            measurementsUnorderedWithClutter = [perfectMeasOrdered(:,[2,3]), [2;2], perfectMeasOrdered(:,1)];
            trackerWithGating.updateLinear(measurementsUnorderedWithClutter, testCase.measMat2DCV, eye(2), 0, zeros(2), 1, true);
            trackerNoGating.updateLinear(measurementsUnorderedWithClutter, testCase.measMat2DCV, eye(2), 0, zeros(2), 1, false);
            testCase.verifyEqual(trackerWithGating.x, trackerNoGating.x);
            testCase.verifyEqual(trackerWithGating.C, trackerNoGating.C);
            
            trackerNoGatingBackup = trackerNoGating.copy();
            % Should not behave identically when outside of gating region
            measurementsUnorderedWithClutter = [perfectMeasOrdered(:,[2,3,1]), [10;10]];
            % When one uses gating, the result is no longer the same as
            % without gating.
            trackerWithGating.updateLinear(measurementsUnorderedWithClutter, testCase.measMat2DCV, eye(2), 0, zeros(2), 1, true);
            trackerNoGating.updateLinear(measurementsUnorderedWithClutter, testCase.measMat2DCV, eye(2), 0, zeros(2), 1, false);
            testCase.verifyNotEqual(trackerWithGating.x, trackerNoGating.x);
            testCase.verifyNotEqual(trackerWithGating.C, trackerNoGating.C);
            % Intead, it should be the same as without the clutter
            % measurement.
            trackerNoGating = trackerNoGatingBackup;
            trackerNoGatingBackup.updateLinear(perfectMeasOrdered(:,[2,3,1]), testCase.measMat2DCV, eye(2), 0, zeros(2), 1, false);
            testCase.verifyEqual(trackerWithGating.x, trackerNoGating.x);
            testCase.verifyEqual(trackerWithGating.C, trackerNoGating.C);
        end

        function testLogging(testCase)
            import matlab.unittest.constraints.HasNaN
            % Init
            tracker = KernelSMEFilter();
            tracker.setState(testCase.gaussians2DCV);
            testCase.verifySize(tracker.priorEstimatesOverTime, [12,1]);
            testCase.verifyEmpty(tracker.posteriorEstimatesOverTime);
      
            % In the first time step, there should be no padding yet.
            % Hence, there should be no NaN values.
            testCase.verifyThat(tracker.priorEstimatesOverTime, ~HasNaN);
            % Update
            allGaussians = [testCase.gaussians2DCV];
            perfectMeasOrdered = testCase.measMat2DCV * [allGaussians.mu];
            measurements = perfectMeasOrdered;
            tracker.updateLinear(measurements, testCase.measMat2DCV, eye(2));
            
            testCase.verifySize(tracker.priorEstimatesOverTime, [12,1]);
            testCase.verifySize(tracker.posteriorEstimatesOverTime, [12,1]);
            % We know there are no NaNs
            testCase.verifyThat(tracker.posteriorEstimatesOverTime, ~HasNaN);
            % Predict
            tracker.predictLinear(blkdiag([1,1;0,1], [1,1;0,1]), eye(4), []);

            testCase.verifySize(tracker.priorEstimatesOverTime, [12,2]);
            testCase.verifySize(tracker.posteriorEstimatesOverTime, [12,1]);
            % We know there are no NaNs
            testCase.verifyThat(tracker.priorEstimatesOverTime, ~HasNaN);
            testCase.verifyThat(tracker.priorEstimatesOverTime, ~HasNaN);

            allGaussians = [testCase.gaussians2DCV];
            % Generate perfect measurements, association should then be
            % optimal.
            perfectMeasOrdered = testCase.measMat2DCV * [allGaussians.mu];
            measurements = perfectMeasOrdered;
            tracker.updateLinear(measurements, testCase.measMat2DCV, eye(2));

            testCase.verifySize(tracker.priorEstimatesOverTime, [12,2]);
            testCase.verifySize(tracker.posteriorEstimatesOverTime, [12,2]);
            testCase.verifySize(tracker.priorEstimatesOverTime, [12,2]);
            testCase.verifySize(tracker.posteriorEstimatesOverTime, [12,2]);
            testCase.verifyThat(tracker.priorEstimatesOverTime, ~HasNaN);
            testCase.verifyThat(tracker.priorEstimatesOverTime, ~HasNaN);
        end
    end
end
