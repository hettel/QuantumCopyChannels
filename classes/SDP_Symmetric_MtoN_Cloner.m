classdef SDP_Symmetric_MtoN_Cloner
    properties
        Omega
        M  % Number of input copies
        N  % Number of output copies
        d  % Dimension of single system (assumed to be 2 for qubits)
    end
    
    methods
        %% Constructor
        function obj = SDP_Symmetric_MtoN_Cloner(M, N, states)
            arguments
                M (1,1) {mustBePositive, mustBeInteger}
                N (1,1) {mustBePositive, mustBeInteger}
                states
            end
            
            % Validate that N >= M (can't clone to fewer copies)
            if N < M
                error('N must be >= M (cannot clone to fewer copies)');
            end
            
            obj.M = M;
            obj.N = N;
            obj.d = 2;  % Qubit dimension
            
            % Calculate dimensions
            dim_in = obj.d^M;
            dim_out = obj.d^N;
            total_dim = dim_in * dim_out;
            
            tmpOmega = zeros(total_dim, total_dim);
            
            for state = states
                rho = state * state'; % Density matrix for pure state
                
                part_in  = Tensor( conj(rho), obj.M );
                part_out = kron( rho, eye( obj.d^(obj.N-1) ) );
                
                tmpOmega = tmpOmega + kron(part_in, part_out);
            end
            
            % Normalize
            obj.Omega = tmpOmega / length(states);
        end
        
        %% Solve the primal SDP problem
        function [primal_optimum, E] = solvePrimal(obj)
            dim_in = obj.d^obj.M;
            dim_out = obj.d^obj.N;
            total_dim = dim_in * dim_out;
            
            cvx_begin sdp quiet
                variable E(total_dim, total_dim) hermitian
                
                % Objective: maximize entanglement fidelity
                maximize( real(trace(E * obj.Omega)) )
                
                subject to
                    % 1. Complete positivity
                    E >= 0;
                    
                    % 2. Trace-preserving condition
                    % Trace out all output subsystems (M+1 to M+N)
                    output_systems = (obj.M+1):(obj.M+obj.N);
                    all_dims = obj.d * ones(1, obj.M + obj.N);

                    E_traced = PartialTrace(E, output_systems, all_dims);
                    E_traced == eye(dim_in);
                    
                    % 3. Symmetry: all permutations of output subsystems
                    % Generate all necessary swap permutations
                    dims = obj.d * ones(1, obj.M + obj.N);
                    
                    % For symmetric cloner, we need output systems to be symmetric
                    % Generate pairwise swaps of output subsystems
                    for i = 2:obj.N
                        perm = 1:(obj.M+obj.N);                      
                        tmp = perm(obj.M + i);
                        perm(obj.M + i) = perm(obj.M + 1);
                        perm(obj.M+1) = tmp;
             
                        E == PermuteSystems(E, perm, dims);
                    end
            cvx_end
            
            primal_optimum = cvx_optval;
        end
        
        %% Solve the dual SDP problem
        function [dual_optimum, Y] = solveDual(obj)
        
            dim_in = obj.d^obj.M;
            dim_out = obj.d^obj.N;
            total_dim = dim_in * dim_out;
        
            num_swaps = obj.N - 1;
            dims = obj.d * ones(1, obj.M + obj.N);
        
            cvx_begin sdp quiet
                variable Y(dim_in, dim_in) hermitian
                variable Z(total_dim, total_dim, num_swaps) hermitian
        
                minimize( real(trace(Y)) )
        
                subject to
                    dual_lhs = kron(Y, eye(dim_out));
        
                    for k = 1:num_swaps
                        out_idx = k + 1;
        
                        % Build permutation vector
                        perm = 1:(obj.M + obj.N);
                        tmp = perm(obj.M + out_idx);
                        perm(obj.M + out_idx) = perm(obj.M + 1);
                        perm(obj.M + 1) = tmp;
        
                        % Permuted version
                        C = PermuteSystems(Z(:,:,k), perm, dims);
        
                        % Add contribution
                        dual_lhs = dual_lhs + (Z(:,:,k) - C);
                    end
        
                    dual_lhs >= obj.Omega;
            cvx_end
        
            dual_optimum = cvx_optval;
        end
        
        %% Helper method to display cloner information
        function info(obj)
            fprintf('Symmetric %d->%d Quantum Cloner\n', obj.M, obj.N);
            fprintf('Input dimension:  %d (= %d^%d)\n', obj.d^obj.M, obj.d, obj.M);
            fprintf('Output dimension: %d (= %d^%d)\n', obj.d^obj.N, obj.d, obj.N);
            fprintf('Total Choi matrix size: %d x %d\n', size(obj.Omega, 1), size(obj.Omega, 2));
        end
    end
end