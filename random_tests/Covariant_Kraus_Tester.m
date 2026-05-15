classdef Covariant_Kraus_Tester
    methods (Static)
        function testKrausOperators(kraus_ops, M, N, F_theoretical, num_tests, epsilon)
            arguments
                kraus_ops  % Kraus operators
                M (1,1) {mustBePositive, mustBeInteger}
                N (1,1) {mustBePositive, mustBeInteger}
                F_theoretical (1,1) double  % Expected fidelity
                num_tests (1,1) {mustBePositive, mustBeInteger} = 10
                epsilon (1,1) double = 1e-8
            end
            
            fprintf('\n=== Testing %d->%d Kraus Cloning ===\n', M, N);
            fprintf('Number of Kraus operators: %d\n', length(kraus_ops));


             % Track statistics
            num_success = 0;
            max_error = 0;
            for test_idx = 1:num_tests
                % create random input state 
                state_in = Utils.create_random_phase_covariant_qubit();
                rho_in   = state_in*state_in';
 
                % Apply channel
                rho_input  = Tensor( rho_in, M);
                rho_out = Utils.applyChannel(kraus_ops, rho_input);

                dims       = 2 * ones(1, N);
                fidelities = zeros(0,N);
                for k=1:N
                    sys_to_trace = setdiff(1:N, k);
                    rho_out_k = PartialTrace(rho_out, sys_to_trace, dims);
                    fidelities(k) = trace(rho_out_k * rho_in);  % Calculate fidelity
                    delta = abs( fidelities(k) - F_theoretical );
                    max_error = max( delta, max_error );
                end

                if max_error < epsilon
                    num_success = num_success + 1;  % Increment success count
                    fprintf("Success\n");
                else
                    fprintf('Test %2d: Error - Fidelities: ', test_idx);
                    fprintf('%12.10f ', fidelities);
                    fprintf('\n');
                end
            end
             % Summary
            fprintf('\n--- Test Summary ---\n');
            fprintf('Passed: %d/%d tests\n', num_success, num_tests);
            fprintf('Success rate: %.1f%%\n', 100*num_success/num_tests);
            fprintf('Maximum deviation: %.2e\n', max_error);
            fprintf('Theoretical fidelity: %.10f\n', F_theoretical);
        end
    end
end