classdef Universal_Cloner_Tester
    methods (Static)
        %% Main test function for M->N cloner
        function testCloner(J, M, N, F_theoretical, num_tests, epsilon)
            arguments
                J                           % Choi matrix
                M (1,1) {mustBePositive, mustBeInteger}
                N (1,1) {mustBePositive, mustBeInteger}
                F_theoretical (1,1) double  % Expected fidelity
                num_tests (1,1) {mustBePositive, mustBeInteger} = 10
                epsilon (1,1) double = 1e-8
            end
            
            fprintf('\n=== Testing %d->%d Cloner ===\n', M, N);
            
            % Get Kraus operators
            dim_in = 2^M;
            dim_out = 2^N;
            kraus_ops = Utils.getKrausOperators(J, M, N);
            fprintf('Number of Kraus operators: %d\n', length(kraus_ops));
            
            % Track statistics
            num_success = 0;
            max_error = 0;
            
            % Run tests
            for test_idx = 1:num_tests
                % Generate random input state
                rho_in = Universal_Cloner_Tester.createRandomInputState(M);
                
                % Get single qubit target state (for fidelity comparison)
                rho_single_target = Universal_Cloner_Tester.getSingleQubitTarget(rho_in, M);
                
                % Apply channel
                rho_out = Utils.applyChannel(kraus_ops, rho_in);
                
                % Test all output clones
                [success, fidelities, max_delta] = Universal_Cloner_Tester.testAllClones(...
                    rho_out, rho_single_target, N, F_theoretical, epsilon);
                
                % Update statistics
                if success
                    num_success = num_success + 1;
                    fprintf('Test %2d: Success\n', test_idx);
                else
                    fprintf('Test %2d: Error - Fidelities: ', test_idx);
                    fprintf('%12.10f ', fidelities);
                    fprintf('\n');
                end
                
                max_error = max(max_error, max_delta);
            end
            
            % Summary
            fprintf('\n--- Test Summary ---\n');
            fprintf('Passed: %d/%d tests\n', num_success, num_tests);
            fprintf('Success rate: %.1f%% (Tolerance %.1e)\n', 100*num_success/num_tests, epsilon);
            fprintf('Maximum deviation: %.2e\n', max_error);
            fprintf('Theoretical fidelity: %.10f\n', F_theoretical);
        end
        
        %% Create random input state for M qubits
        function rho_in = createRandomInputState(M)
            if M == 1
                % Single qubit case
                state_in = Universal_Cloner_Tester.create_random_qubit();
                rho_in = state_in * state_in';
            else
                % Multiple qubits case
                state_init = [1; 0];
                
                % Random Haar measure rotation
                lambda = rand * 2 * pi;
                theta = rand * pi;
                phi = rand * 2 * pi;
                
                U = [cos(theta/2), -exp(1i*lambda)*sin(theta/2);
                     exp(1i*phi)*sin(theta/2), exp(1i*(phi+lambda))*cos(theta/2)];
                
                state_in = U * state_init;
                rho_single = state_in * state_in';
                
                % Create M copies (tensor product)
                rho_in = Tensor(rho_single, M);
            end
        end
        
        %% Extract single qubit target state from input
        function rho_single = getSingleQubitTarget(rho_in, M)
            if M == 1
                rho_single = rho_in;
            else
                % Trace out all but first qubit
                systems_to_trace = 2:M;
                dims = 2 * ones(1, M);
                rho_single = PartialTrace(rho_in, systems_to_trace, dims);
            end
        end
        
        %% Test all output clones for symmetry and fidelity
        function [success, fidelities, max_delta] = testAllClones(...
                rho_out, rho_single_target, N, F_theoretical, epsilon)
            
            dims = 2 * ones(1, N);
            fidelities = zeros(1, N);
            deltas = zeros(1, N);
            
            % Test each output clone
            for clone_idx = 1:N
                % Systems to trace out (all except current clone)
                systems_to_trace = setdiff(1:N, clone_idx);
                
                % Partial trace to get single clone
                rho_clone = PartialTrace(rho_out, systems_to_trace, dims);
                
                % Calculate fidelity
                fidelities(clone_idx) = real(trace(rho_single_target * rho_clone));
                deltas(clone_idx) = abs(F_theoretical - fidelities(clone_idx));
            end
            
            % Check if all clones pass
            max_delta = max(deltas);
            success = all(deltas <= epsilon);
        end
        
        %% Create random qubit (uniform Haar measure)
        function state = create_random_qubit()
            % Random parameters for Bloch sphere
            theta = acos(2*rand - 1);  % Uniform on sphere
            phi = 2 * pi * rand;
            
            % Qubit state
            state = [cos(theta/2); exp(1i*phi)*sin(theta/2)];
        end
        
        %% Detailed test with individual clone analysis
        function testClonerDetailed(J, M, N, F_theoretical, epsilon)
            arguments
                J
                M (1,1) {mustBePositive, mustBeInteger}
                N (1,1) {mustBePositive, mustBeInteger}
                F_theoretical (1,1) double
                epsilon (1,1) double = 1e-8
            end
            
            fprintf('\n=== Detailed Test: %d->%d Cloner ===\n', M, N);
            
            % Get Kraus operators
            dim_in = 2^M;
            dim_out = 2^N;
            kraus_ops = Utils.getKrausOperators(J, dim_in, dim_out);
            
            % Single test with detailed output
            rho_in = ClonerTester.createRandomInputState(M);
            rho_single_target = ClonerTester.getSingleQubitTarget(rho_in, M);
            rho_out = Utils.applyChannel(kraus_ops, rho_in);
            
            fprintf('\nInput state purity: %.6f\n', real(trace(rho_in^2)));
            fprintf('Output state purity: %.6f\n', real(trace(rho_out^2)));
            
            dims = 2 * ones(1, N);
            
            fprintf('\nIndividual clone fidelities:\n');
            for clone_idx = 1:N
                systems_to_trace = setdiff(1:N, clone_idx);
                rho_clone = PartialTrace(rho_out, systems_to_trace, dims);
                
                F = real(trace(rho_single_target * rho_clone));
                delta = abs(F_theoretical - F);
                
                status = '✓';
                if delta > epsilon
                    status = '✗';
                end
                
                fprintf('  Clone %d: F = %.10f (Δ = %.2e) %s\n', ...
                    clone_idx, F, delta, status);
            end
            
            % Test symmetry between clones
            fprintf('\nSymmetry check (pairwise clone comparison):\n');
            for i = 1:min(N-1, 3)  % Check first few pairs
                for j = i+1:min(i+1, N)
                    systems_i = setdiff(1:N, i);
                    systems_j = setdiff(1:N, j);
                    rho_i = PartialTrace(rho_out, systems_i, dims);
                    rho_j = PartialTrace(rho_out, systems_j, dims);
                    
                    diff = norm(rho_i - rho_j, 'fro');
                    fprintf('  ||ρ_%d - ρ_%d|| = %.2e\n', i, j, diff);
                end
            end
        end
        
        %% Benchmark test with statistics
        function stats = benchmarkCloner(J, M, N, F_theoretical, num_tests)
            arguments
                J
                M (1,1) {mustBePositive, mustBeInteger}
                N (1,1) {mustBePositive, mustBeInteger}
                F_theoretical (1,1) double
                num_tests (1,1) {mustBePositive, mustBeInteger} = 100
            end
            
            dim_in = 2^M;
            dim_out = 2^N;
            kraus_ops = Utils.getKrausOperators(J, dim_in, dim_out);
            
            all_fidelities = zeros(num_tests, N);
            dims = 2 * ones(1, N);
            
            fprintf('Running %d benchmark tests...\n', num_tests);
            
            for test_idx = 1:num_tests
                rho_in = ClonerTester.createRandomInputState(M);
                rho_single_target = ClonerTester.getSingleQubitTarget(rho_in, M);
                rho_out = Utils.applyChannel(kraus_ops, rho_in);
                
                for clone_idx = 1:N
                    systems_to_trace = setdiff(1:N, clone_idx);
                    rho_clone = PartialTrace(rho_out, systems_to_trace, dims);
                    all_fidelities(test_idx, clone_idx) = ...
                        real(trace(rho_single_target * rho_clone));
                end
                
                if mod(test_idx, 20) == 0
                    fprintf('  Progress: %d/%d\n', test_idx, num_tests);
                end
            end
            
            % Calculate statistics
            stats.mean_fidelity = mean(all_fidelities, 'all');
            stats.std_fidelity = std(all_fidelities, 0, 'all');
            stats.min_fidelity = min(all_fidelities, [], 'all');
            stats.max_fidelity = max(all_fidelities, [], 'all');
            stats.mean_deviation = abs(stats.mean_fidelity - F_theoretical);
            
            fprintf('\n--- Benchmark Results ---\n');
            fprintf('Theoretical fidelity: %.10f\n', F_theoretical);
            fprintf('Mean fidelity:        %.10f ± %.2e\n', ...
                stats.mean_fidelity, stats.std_fidelity);
            fprintf('Range:                [%.10f, %.10f]\n', ...
                stats.min_fidelity, stats.max_fidelity);
            fprintf('Mean deviation:       %.2e\n', stats.mean_deviation);
        end
    end
end