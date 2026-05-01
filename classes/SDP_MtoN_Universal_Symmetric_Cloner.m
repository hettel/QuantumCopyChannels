classdef SDP_MtoN_Universal_Symmetric_Cloner
    % SDP_MtoN_Cloning - Semidefinite programming for optimal M->N quantum cloning
    %
    % This class implements the optimization problem for finding the optimal
    % quantum cloning map that transforms M input qubits into N output qubits
    % with maximum fidelity, subject to complete positivity, trace preservation,
    % and permutation symmetry constraints.
    %
    % Properties:
    %   States - Set of quantum states used to construct Omega
    %   Omega  - Integral operator for fidelity computation
    %   M      - Number of input copies
    %   N      - Number of output clones
    %
    % Methods:
    %   SDP_MtoN_Cloning      - Constructor
    %   solvePrimal           - Solve the primal optimization problem
    %   solveDual             - Solve the dual optimization problem
    %   getKrausOperators     - Extract Kraus operators from Choi matrix
    %   verifyKrausOperators  - Verify completeness of Kraus operators
    %   applyChannel          - Apply quantum channel to an input state
    %   verifyPrimal          - Verify that a solution satisfies all constraints
    %
    % Example:
    %   % Generate random states
    %   states = randn(2, 100) + 1i*randn(2, 100);
    %   for i = 1:100
    %       states(:,i) = states(:,i) / norm(states(:,i));
    %   end
    %
    %   % Create M->N cloner (e.g., 2->5 cloning)
    %   cloner = SDP_MtoN_Cloning(states, 2, 5);
    %   [fidelity, E] = cloner.solvePrimal();
    %
    %   % Verify solution
    %   [is_valid, violations] = cloner.verifyPrimal(E);
    %
    %   % Extract Kraus operators
    %   [kraus_ops, eigenvalues] = cloner.getKrausOperators(E);
    
    properties
        Omega   % Integral operator for fidelity computation (hermitian matrix)
        M       % Number of input copies (positive integer)
        N       % Number of output clones (positive integer, N ≥ M)
    end
    
    methods
        function obj = SDP_MtoN_Universal_Symmetric_Cloner( M, N)
            % SDP_MtoN_Cloning - Construct an M->N quantum cloning optimizer
            %
            % SYNTAX:
            %   obj = SDP_MtoN_Cloning(states, M, N)
            %
            % INPUT:
            %   states - Array of quantum states (2 x num_states)
            %            Each column is a normalized qubit state |ψ⟩
            %   M      - Number of input copies (positive integer)
            %   N      - Number of output clones (positive integer, N ≥ M)
            %
            % OUTPUT:
            %   obj    - SDP_MtoN_Cloning object with initialized Omega matrix
            %
            % DESCRIPTION:
            %   Constructs the Omega matrix used in the fidelity optimization:
            %   Ω = (1/num_states) * Σ_ψ |ψ*⟩⟨ψ*|^⊗M ⊗ |ψ⟩⟨ψ|^⊗N
            %
            %   The Omega matrix has dimension 2^(M+N) × 2^(M+N) and represents
            %   the integral over the Haar measure for universal cloning.
            %
            %   Structure: Input*^⊗M ⊗ Output^⊗N
            %            = In*_1 ⊗ ... ⊗ In*_M ⊗ Out_1 ⊗ ... ⊗ Out_N
            %
            % CONSTRAINT:
            %   N must be ≥ M (cannot clone fewer copies than inputs)
            %
            % NOTE:
            %   For large M+N, the matrix dimension 2^(M+N) can become very large.
            %   A warning is issued for dim_total > 256.
            %
            % EXAMPLE:
            %   % 2->3 cloning (2 inputs to 3 outputs)
            %   cloner = SDP_MtoN_Cloning(states, 2, 3);
            
            arguments
                M {mustBeInteger, mustBePositive}
                N {mustBeInteger, mustBePositive}
            end
            
            % Validate N >= M
            if N < M
                error('Number of outputs N must be >= number of inputs M. Got M=%d, N=%d', M, N);
            end
            
            obj.M = M;
            obj.N = N;
            
            % Dimension: 2^M (inputs*) ⊗ 2^N (N outputs)
            dim_in = 2^M;
            dim_out = 2^N;
            dim_total = dim_in * dim_out;
            
            if dim_total > 256
                warning('Large dimension %d detected (M=%d, N=%d). Computation may be slow or fail.', ...
                        dim_total, M, N);
            end

            % symmetric informationally complete states with redundance
            s0  = [1;0];
            s1  = [0;1];
            sp  = [1;1]/sqrt(2);
            sm  = [1;-1]/sqrt(2);
            spi = [1;1i]/sqrt(2);
            smi = [1;-1i]/sqrt(2);
            states = [s0, s1, sp, sm, spi, smi ];

            tmpOmega = zeros(dim_total, dim_total);
            
            for state = states
                rho = state * state';
                
                % Omega = ∫ |ψ*⟩⟨ψ*|^⊗M ⊗ |ψ⟩⟨ψ|^⊗N dψ
                % Structure: Input*_1 ⊗ ... ⊗ Input*_M ⊗ Output_1 ⊗ ... ⊗ Output_N
                
                % Start with M copies of conjugated input
                term = conj(rho);
                for i = 2:M
                    term = kron(term, conj(rho));
                end
                
                term = kron(term, rho);
                % Add N copies of output
                for i = 2:N
                    term = kron(term, eye(2));
                end
                
                tmpOmega = tmpOmega + term;
            end
            
            % Average over all states
            obj.Omega = tmpOmega / length(states);
        end
        
        function [primal_optimum, E] = solvePrimal(obj)
            % solvePrimal - Solve the primal SDP for optimal M->N cloning fidelity
            %
            % SYNTAX:
            %   [optimum, E] = solvePrimal(obj)
            %
            % INPUT:
            %   obj - SDP_MtoN_Cloning object
            %
            % OUTPUT:
            %   optimum - Optimal fidelity value (scalar in [0,1])
            %   E       - Optimal Choi matrix (dim_total × dim_total hermitian)
            %
            % DESCRIPTION:
            %   Solves the primal semidefinite program:
            %
            %   maximize    Tr(E * Ω)
            %   subject to  E ≥ 0                    (Complete Positivity)
            %               Tr_out(E) = I_in         (Trace Preserving)
            %               P_in(E) = E              (Input Permutation Symmetry)
            %               P_out(E) = E             (Output Permutation Symmetry)
            %
            %   where P_in permutes input systems and P_out permutes output systems.
            %
            %   The objective Tr(E * Ω) represents the average fidelity of
            %   the cloning operation over all input states.
            %
            % CONSTRAINTS:
            %   1. CP (Complete Positivity): E ≥ 0
            %      The Choi matrix must be positive semidefinite
            %
            %   2. TP (Trace Preserving): Tr_{out1,...,outN}(E) = I_{in1,...,inM}
            %      Tracing out all N output systems yields the M-qubit input identity
            %
            %   3. Input Symmetry: E = P_{i,j}^in(E) for all i,j ∈ {1,...,M}
            %      The map is invariant under permutations of input systems
            %
            %   4. Output Symmetry: E = P_{i,j}^out(E) for all i,j ∈ {1,...,N}
            %      The map is invariant under permutations of output systems
            %
            % SOLVER:
            %   Uses CVX with default solver (SDPT3, SeDuMi, or MOSEK)
            %
            % EXAMPLE:
            %   cloner = SDP_MtoN_Cloning(states, 2, 4);
            %   [fidelity, E] = cloner.solvePrimal();
            %   fprintf('Optimal fidelity (2->4): %.6f\n', fidelity);
            %
            % SEE ALSO:
            %   solveDual, verifyPrimal, theoreticalBound
            
            dim_in = 2^obj.M;
            dim_out = 2^obj.N;
            dim_total = dim_in * dim_out;
            
            cvx_begin sdp quiet
                variable E(dim_total, dim_total) hermitian
                
                % Objective function: Maximize fidelity
                maximize( real(trace(E * obj.Omega)) )
                
                subject to
                    % 1. Completely Positive (CP)
                    0.5 * (E + E') >= 0;  % E >= 0
                    
                    % 2. Trace Preserving (TP): Tr_{out1,...,outN}(E) = I_{in}
                    % Trace out all N output systems (systems M+1 to M+N)
                    dims = 2 * ones(1, obj.M + obj.N);
                    PartialTrace(E, (obj.M+1):(obj.M+obj.N), dims) == eye(dim_in);
                    
                    % 3. Input Symmetry: Invariance under permutations of M input systems
                    %    Adjacent transpositions generate the full symmetric group S_M
                    for i = 1:(obj.M-1)
                        % Swap input systems i and i+1
                        perm = 1:(obj.M + obj.N);
                        perm(i) = i+1;
                        perm(i+1) = i;
                        
                        E == PermuteSystems(E, perm, dims);
                    end
                    
                    % 4. Output Symmetry: Invariance under permutations of N output systems
                    %    Adjacent transpositions generate the full symmetric group S_N
                    for i = (obj.M+1):(obj.M+obj.N-1)
                        % Swap output systems i and i+1
                        perm = 1:(obj.M + obj.N);
                        perm(i) = i+1;
                        perm(i+1) = i;
                        
                        E == PermuteSystems(E, perm, dims);
                    end
            cvx_end
            
            primal_optimum = cvx_optval;
        end
        
        function [dual_optimum, Y] = solveDual(obj)
            % solveDual - Solve the dual SDP for optimal M->N cloning fidelity
            %
            % SYNTAX:
            %   [optimum, Y] = solveDual(obj)
            %
            % INPUT:
            %   obj - SDP_MtoN_Cloning object
            %
            % OUTPUT:
            %   optimum - Optimal dual objective value (should equal primal)
            %   Y       - Optimal dual variable for TP constraint (2^M × 2^M hermitian)
            %
            % DESCRIPTION:
            %   Solves the dual semidefinite program:
            %
            %   minimize    Tr(Y)
            %   subject to  Y ⊗ I + Σ_i Z^in_i + Σ_j Z^out_j ≥ Ω
            %
            %   where:
            %   - Y is the dual variable for the TP constraint
            %   - Z^in_i are dual variables for input symmetry constraints
            %   - Z^out_j are dual variables for output symmetry constraints
            %
            % DUAL VARIABLES:
            %   Y       - 2^M × 2^M hermitian matrix (TP constraint)
            %   Z^in_i  - dim_total × dim_total hermitian (M-1 input symmetries)
            %   Z^out_j - dim_total × dim_total hermitian (N-1 output symmetries)
            %
            % STRONG DUALITY:
            %   Under Slater's condition (which holds here), strong duality
            %   guarantees that the primal and dual optimal values are equal.
            %
            % SEE ALSO:
            %   solvePrimal
            
            dim_in = 2^obj.M;
            dim_out = 2^obj.N;
            dim_total = dim_in * dim_out;
            dims = 2 * ones(1, obj.M + obj.N);
            
            % Total number of symmetry constraints
            num_sym_in = obj.M - 1;   % Input symmetries
            num_sym_out = obj.N - 1;  % Output symmetries
            num_sym_total = num_sym_in + num_sym_out;
            
            cvx_begin sdp quiet
                variable Y(dim_in, dim_in) hermitian
                
                % Dual variables for symmetry constraints
                if num_sym_total > 0
                    variable Z_all(dim_total, dim_total, num_sym_total) hermitian
                end
                
                minimize( real(trace(Y)) )
                
                subject to
                    % Construct the dual constraint
                    dual_matrix = kron(Y, eye(dim_out));
                    
                    if num_sym_total > 0
                        % Add input symmetry constraint terms
                        for i = 1:num_sym_in
                            Z = Z_all(:,:,i);
                            
                            % Permutation for swapping inputs i and i+1
                            perm = 1:(obj.M + obj.N);
                            perm(i) = i+1;
                            perm(i+1) = i;
                            
                            dual_matrix = dual_matrix + (Z - PermuteSystems(Z, perm, dims));
                        end
                        
                        % Add output symmetry constraint terms
                        for j = 1:num_sym_out
                            Z = Z_all(:,:,num_sym_in + j);
                            
                            % Permutation for swapping outputs j and j+1
                            % (outputs start at position M+1)
                            perm = 1:(obj.M + obj.N);
                            perm(obj.M + j) = obj.M + j + 1;
                            perm(obj.M + j + 1) = obj.M + j;
                            
                            dual_matrix = dual_matrix + (Z - PermuteSystems(Z, perm, dims));
                        end
                    end
                    
                    % Dual constraint
                    dual_matrix >= obj.Omega;
            cvx_end
            
            dual_optimum = cvx_optval;
        end
        
        function [kraus_ops, eigenvalues] = getKrausOperators(obj, E, threshold)
            % getKrausOperators - Extract Kraus operators from Choi matrix
            %
            % SYNTAX:
            %   [kraus_ops, eigenvalues] = getKrausOperators(obj, E)
            %   [kraus_ops, eigenvalues] = getKrausOperators(obj, E, threshold)
            %
            % INPUT:
            %   E         - Choi matrix (dim_total × dim_total hermitian)
            %   threshold - (optional) Eigenvalue cutoff threshold
            %               Default: 1e-10
            %
            % OUTPUT:
            %   kraus_ops   - Cell array of Kraus operators {K_1, K_2, ..., K_r}
            %                 Each K_i is (2^N × 2^M) mapping M inputs to N outputs
            %   eigenvalues - All eigenvalues of E in descending order
            %
            % DESCRIPTION:
            %   Computes Kraus operators via eigenvalue decomposition of the
            %   Choi matrix for M->N cloning.
            %
            % DIMENSIONS:
            %   Input space:  2^M (M qubits)
            %   Output space: 2^N (N qubits)
            %   Each K_i:     2^N × 2^M matrix
            %
            % SEE ALSO:
            %   verifyKrausOperators, applyChannel
            
            arguments
                obj
                E {mustBeNumeric}
                threshold = 1e-10
            end
            
            dim_in = 2^obj.M;      % Input dimension (M qubits)
            dim_out = 2^obj.N;     % Output dimension (N qubits)
            dim_total = dim_in * dim_out;
            
            % Verify matrix dimensions
            if ~isequal(size(E), [dim_total, dim_total])
                error('Choi matrix E has incorrect dimensions. Expected %dx%d, got %dx%d', ...
                      dim_total, dim_total, size(E,1), size(E,2));
            end
            
            % Ensure E is Hermitian
            E = (E + E') / 2;
            
            % Eigenvalue decomposition
            [V, D] = eig(E);
            eigenvalues = diag(D);
            
            % Sort eigenvalues in descending order
            [eigenvalues, idx] = sort(real(eigenvalues), 'descend');
            V = V(:, idx);
            
            % Filter out small/negative eigenvalues
            positive_idx = eigenvalues > threshold;
            eigenvalues_filtered = eigenvalues(positive_idx);
            V_filtered = V(:, positive_idx);
            
            num_kraus = sum(positive_idx);
            
            if num_kraus == 0
                warning('No positive eigenvalues found above threshold %.2e', threshold);
                kraus_ops = {};
                eigenvalues = [];
                return;
            end
            
            % Extract Kraus operators
            kraus_ops = cell(num_kraus, 1);
            
            for i = 1:num_kraus
                % Get eigenvector
                vec_K = V_filtered(:, i);
                
                % Reshape for (in ⊗ out) convention
                K = reshape(vec_K, [dim_out, dim_in]);
                
                % Scale with sqrt of eigenvalue
                K = sqrt(eigenvalues_filtered(i)) * K;
                
                kraus_ops{i} = K;
            end
            
            eigenvalues = eigenvalues;
        end
        
        function [is_valid, completeness_error] = verifyKrausOperators(obj, kraus_ops, tol)
            % verifyKrausOperators - Verify completeness of Kraus operators
            %
            % SYNTAX:
            %   [is_valid, completeness_error] = verifyKrausOperators(obj, kraus_ops)
            %   [is_valid, completeness_error] = verifyKrausOperators(obj, kraus_ops, tol)
            %
            % INPUT:
            %   kraus_ops - Cell array of Kraus operators
            %   tol       - (optional) Tolerance (default: 1e-6)
            %
            % OUTPUT:
            %   is_valid           - Boolean: true if completeness satisfied
            %   completeness_error - ||Σ_i K_i† K_i - I_{2^M}||_F
            %
            % DESCRIPTION:
            %   Verifies the completeness relation for M->N cloning:
            %   Σ_i K_i† K_i = I_{2^M}
            %
            % SEE ALSO:
            %   getKrausOperators, verifyPrimal
            
            arguments
                obj
                kraus_ops {mustBeA(kraus_ops, 'cell')}
                tol = 1e-6
            end
            
            if isempty(kraus_ops)
                is_valid = false;
                completeness_error = inf;
                return;
            end
            
            dim_in = 2^obj.M;  % Input dimension
            
            % Compute sum_i K_i† K_i
            completeness = zeros(dim_in, dim_in);
            
            for i = 1:length(kraus_ops)
                K = kraus_ops{i};
                completeness = completeness + K' * K;
            end
            
            % Check deviation from identity
            completeness_error = norm(completeness - eye(dim_in), 'fro');
            is_valid = completeness_error <= tol;
        end
        
        function rho_out = applyChannel(obj, kraus_ops, rho_in)
            % applyChannel - Apply M->N cloning channel to input state
            %
            % SYNTAX:
            %   rho_out = applyChannel(obj, kraus_ops, rho_in)
            %
            % INPUT:
            %   kraus_ops - Cell array of Kraus operators
            %   rho_in    - Input density matrix (2^M × 2^M hermitian)
            %
            % OUTPUT:
            %   rho_out   - Output density matrix (2^N × 2^N hermitian)
            %
            % DESCRIPTION:
            %   Applies the M->N cloning channel:
            %   ρ_out = Σ_i K_i ρ_in K_i†
            %
            %   Maps M input qubits to N output qubits.
            %
            % EXAMPLE:
            %   % 2->3 cloning of |00⟩ state
            %   psi_00 = kron([1;0], [1;0]);
            %   rho_in = psi_00 * psi_00';
            %   rho_out = cloner.applyChannel(kraus_ops, rho_in);
            %
            % SEE ALSO:
            %   getKrausOperators, verifyKrausOperators
            
            arguments
                obj
                kraus_ops {mustBeA(kraus_ops, 'cell')}
                rho_in {mustBeNumeric}
            end
            
            dim_out = 2^obj.N;
            rho_out = zeros(dim_out, dim_out);
            
            for i = 1:length(kraus_ops)
                K = kraus_ops{i};
                rho_out = rho_out + K * rho_in * K';
            end
        end
        
        function [is_valid, violations] = verifyPrimal(obj, E, tol)
            % verifyPrimal - Verify that Choi matrix satisfies all constraints
            %
            % SYNTAX:
            %   [is_valid, violations] = verifyPrimal(obj, E)
            %   [is_valid, violations] = verifyPrimal(obj, E, tol)
            %
            % INPUT:
            %   E   - Choi matrix to verify
            %   tol - (optional) Tolerance (default: 1e-6)
            %
            % OUTPUT:
            %   is_valid   - Boolean: true if all constraints satisfied
            %   violations - Structure with fields:
            %                .CP            - Minimum eigenvalue of E
            %                .TP            - ||Tr_out(E) - I||_F
            %                .symmetry_in   - Input symmetry violations (M-1 values)
            %                .symmetry_out  - Output symmetry violations (N-1 values)
            %
            % DESCRIPTION:
            %   Verifies all constraints for M->N cloning:
            %   1. CP: E ≥ 0
            %   2. TP: Tr_out(E) = I_{in}
            %   3. Input symmetry: P_i^in(E) = E for i = 1,...,M-1
            %   4. Output symmetry: P_j^out(E) = E for j = 1,...,N-1
            %
            % SEE ALSO:
            %   solvePrimal, verifyKrausOperators
            
            arguments
                obj
                E {mustBeNumeric}
                tol = 1e-6
            end
            
            violations = struct();
            dims = 2 * ones(1, obj.M + obj.N);
            dim_in = 2^obj.M;
            
            % Check CP
            eigs_E = eig(E);
            violations.CP = min(eigs_E);
            
            % Check TP
            TP_result = PartialTrace(E, (obj.M+1):(obj.M+obj.N), dims);
            violations.TP = norm(TP_result - eye(dim_in), 'fro');
            
            % Check input symmetry for all adjacent swaps
            violations.symmetry_in = zeros(obj.M-1, 1);
            for i = 1:(obj.M-1)
                perm = 1:(obj.M + obj.N);
                perm(i) = i+1;
                perm(i+1) = i;
                
                E_perm = PermuteSystems(E, perm, dims);
                violations.symmetry_in(i) = norm(E - E_perm, 'fro');
            end
            
            % Check output symmetry for all adjacent swaps
            violations.symmetry_out = zeros(obj.N-1, 1);
            for j = 1:(obj.N-1)
                perm = 1:(obj.M + obj.N);
                perm(obj.M + j) = obj.M + j + 1;
                perm(obj.M + j + 1) = obj.M + j;
                
                E_perm = PermuteSystems(E, perm, dims);
                violations.symmetry_out(j) = norm(E - E_perm, 'fro');
            end
            
            % Overall validity check
            is_valid = violations.CP >= -tol && ...
                       violations.TP <= tol && ...
                       all(violations.symmetry_in <= tol) && ...
                       all(violations.symmetry_out <= tol);
        end
        
        function bound = theoreticalBound(obj, d)
            % theoreticalBound - Compute theoretical upper bound on fidelity
            %
            % SYNTAX:
            %   bound = theoreticalBound(obj)
            %   bound = theoreticalBound(obj, d)
            %
            % INPUT:
            %   d - (optional) Dimension of quantum system (default: 2 for qubits)
            %
            % OUTPUT:
            %   bound - Theoretical optimal fidelity for M->N cloning
            %
            % DESCRIPTION:
            %   Computes the theoretical upper bound using the formula:
            %
            %   F_opt = (M*N + d) / (M*(N + d))
            %
            %   where:
            %   - M = number of input copies
            %   - N = number of output copies
            %   - d = dimension (2 for qubits)
            %
            % FORMULA FOR QUBITS (d=2):
            %   F_opt = (M*N + 2) / (M*(N + 2))
            %
            % EXAMPLES:
            %   M=1, N=2:  F = 4/6 = 2/3     ≈ 0.6667
            %   M=2, N=3:  F = 8/10 = 4/5    = 0.8
            %   M=2, N=4:  F = 10/12 = 5/6   ≈ 0.8333
            %   M=3, N=5:  F = 17/21         ≈ 0.8095
            %
            % LIMITING CASES:
            %   M=N:     F = (N^2 + 2)/(N^2 + 2N) → 1 as N→∞ (no cloning needed)
            %   N>>M:    F → M/(M+2)              (independent of N for large N)
            %
            % REFERENCE:
            %   N. Gisin and S. Massar, Phys. Rev. Lett. 79, 2153 (1997)
            %
            % USAGE:
            %   [fid_num, E] = cloner.solvePrimal();
            %   fid_theory = cloner.theoreticalBound();
            %   fprintf('Numerical:   %.8f\n', fid_num);
            %   fprintf('Theoretical: %.8f\n', fid_theory);
            %   fprintf('Gap:         %.2e\n', abs(fid_num - fid_theory));
            %
            % SEE ALSO:
            %   solvePrimal, verifyPrimal
            
            arguments
                obj
                d = 2  % Qubit dimension
            end
            
            % Formula: F_opt = (M*N + d) / (M*(N + d))
            bound = (obj.M * obj.N + d) / (obj.M * (obj.N + d));
        end
    end
end