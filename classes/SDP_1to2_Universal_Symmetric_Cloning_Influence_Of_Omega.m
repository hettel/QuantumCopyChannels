classdef SDP_1to2_Universal_Symmetric_Cloning_Influence_Of_Omega
    % SDP_1toN_Cloning - Semidefinite programming for optimal 1->N quantum cloning
    %
    % This class implements the optimization problem for finding the optimal
    % quantum cloning map that transforms 1 input qubit into N
    % output qubits with maximum fidelity, subject to complete positivity,
    % trace preservation, and permutation symmetry constraints.
    %
    % Properties:
    %   States - Set of quantum states used to construct Omega
    %   Omega  - Integral operator ∫ |ψ*⟩⟨ψ*| ⊗ |ψ⟩⟨ψ|^⊗N dψ
    %   N      - Number of output clones
    %
    % Methods:
    %   SDP_1toN_Cloning  - Constructor
    %   solvePrimal       - Solve the primal optimization problem
    %   solveDual         - Solve the dual optimization problem
    %   getKrausOperators - Extract Kraus operators from Choi matrix
    %   verifyKrausOperators - Verify completeness of Kraus operators
    %   applyChannel      - Apply quantum channel to an input state
    %   verifyPrimal      - Verify that a solution satisfies all constraints
    %
    % Example:
    %   % Generate random states
    %   states = randn(2, 100) + 1i*randn(2, 100);
    %   for i = 1:100
    %       states(:,i) = states(:,i) / norm(states(:,i));
    %   end
    %
    %   % Create cloner and solve
    %   cloner = SDP_1toN_Cloning(states);
    %   [fidelity, E] = cloner.solvePrimal();
    %
    %   % Verify solution
    %   [is_valid, violations] = cloner.verifyPrimal(E);
    %
    %   % Extract Kraus operators
    %   [kraus_ops, eigenvalues] = cloner.getKrausOperators(E);
    
    properties
        States  % Set of quantum states (2 x num_states array)
        Omega   % Integral operator for fidelity computation (hermitian matrix)
        N {mustBeInteger, mustBePositive}
    end
    
    methods
        function obj = SDP_1to2_Universal_Symmetric_Cloning_Influence_Of_Omega(states)
            % SDP_1toN_Cloning - Construct a 1->N quantum cloning optimizer
            %
            % SYNTAX:
            %   obj = SDP_1toN_Cloning(states)
            %
            % INPUT:
            %   states - Array of quantum states (2 x num_states)
            %            Each column is a normalized qubit state |ψ⟩
            %
            % OUTPUT:
            %   obj    - Object with initialized Omega matrix
            %
            % DESCRIPTION:
            %   Constructs the Omega matrix used in the fidelity optimization:
            %   Ω = (1/num_states) * Σ_ψ |ψ*⟩⟨ψ*| ⊗ |ψ⟩⟨ψ| ⊗ ... ⊗ |ψ⟩⟨ψ|
            %
            %   The Omega matrix has dimension 2^(N+1) × 2^(N+1) and represents
            %   the integral over the Haar measure for universal cloning when
            %   using a sufficiently large set of random states.
            %
            %   Structure: Input* ⊗ Output1 ⊗ Output2 ⊗ ... ⊗ OutputN
            %
            % NOTE:
            %   For large N (N ≥ 8), the matrix dimension becomes 2^(N+1) ≥ 512,
            %   which may lead to memory or numerical issues. A warning is
            %   issued for dim_total > 256.
            
            arguments
                states {mustBeNumeric}
            end
            
            obj.States = states;
            obj.N = 2;
            
            dim_total = 4;
            
            if dim_total > 256
                warning('Large dimension %d detected. Computation may be slow or fail.', dim_total);
            end

            tmpOmega = zeros(dim_total, dim_total);
            
            for state = states
                rho = state * state';
                
                % Omega = ∫ |ψ*⟩⟨ψ*| ⊗ |ψ⟩⟨ψ| ⊗ ... ⊗ |ψ⟩⟨ψ|  dψ
                % Input* ⊗ Output1 ⊗ Output2 ⊗ ... ⊗ OutputN
                
                % Start with conjugated input
                term = conj(rho);
                term = kron(term, rho);
                
                tmpOmega = tmpOmega + term;
            end

            % Tensor product with N-1 copies of 1
            for i = 1:(obj.N-1)
                tmpOmega = kron(tmpOmega, eye(2));
            end
                        
            % Average over all states
            obj.Omega = tmpOmega / length(states);
        end
        
        function [optimum, E] = solvePrimal(obj)
            % solvePrimal - Solve the primal SDP for optimal cloning fidelity
            %
            % SYNTAX:
            %   [optimum, E] = solvePrimal(obj)
            %
            % INPUT:
            %   obj - SDP_1toN_Cloning object
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
            %               P_i(E) = E  ∀i           (Permutation Symmetry)
            %
            %   where P_i permutes output systems i and i+1.
            %
            %   The objective Tr(E * Ω) represents the average fidelity of
            %   the cloning operation over all input states.
            %
            % CONSTRAINTS:
            %   1. CP (Complete Positivity): E ≥ 0
            %      The Choi matrix must be positive semidefinite
            %
            %   2. TP (Trace Preserving): Tr_{out1,...,outN}(E) = I_2
            %      Tracing out all output systems yields the input identity
            %
            %   3. Symmetry: E = P_{i,i+1}(E) for all i ∈ {1,...,N-1}
            %      The map is invariant under permutations of output systems
            %      These N-1 adjacent transpositions generate the full
            %      symmetric group S_N
            %
            % SOLVER:
            %   Uses CVX with default solver (SDPT3, SeDuMi, or MOSEK)

            
            dim_total = 2^(obj.N+1);
            
            cvx_begin sdp quiet
                variable E(dim_total, dim_total) hermitian
                
                % Objective function: Maximize fidelity
                maximize( real(trace(E * obj.Omega)) )
                
                subject to
                    % 1. Completely Positive (CP)
                    E >= 0;
                    
                    % 2. Trace Preserving (TP): Tr_{out1,...,outN}(E) = I_in
                    % Trace out all N output systems (systems 2 to N+1)
                    dims = 2 * ones(1, obj.N+1);
                    PartialTrace(E, 2:(obj.N+1), dims) == eye(2);
                    
                    % 3. Symmetry constraints: E must be invariant under
                    %    all permutations of the N output systems
                    %    We enforce this by requiring equality for all
                    %    adjacent transpositions (generates full symmetry group)
                    for i = 2:obj.N
                        % Swap output systems i and i+1
                        % System indices: [1, 2, 3, ..., N+1]
                        % Swap positions i and i+1 (in 1-indexed system list)
                        perm = 1:(obj.N+1);
                        perm(i) = i+1;
                        perm(i+1) = i;
                        
                        E == PermuteSystems(E, perm, dims);
                    end
            cvx_end
            
            optimum = cvx_optval;
        end
        
        function [optimum, Y] = solveDual(obj)
            % solveDual - Solve the dual SDP for optimal cloning fidelity
            %
            % SYNTAX:
            %   [optimum, Y] = solveDual(obj)
            %
            % INPUT:
            %   obj - SDP_1toN_Cloning object
            %
            % OUTPUT:
            %   optimum - Optimal dual objective value (should equal primal)
            %   Y       - Optimal dual variable for TP constraint (2 × 2 hermitian)
            %
            % DESCRIPTION:
            %   Solves the dual semidefinite program:
            %
            %   minimize    Tr(Y)
            %   subject to  Y ⊗ I + Σ_i (Z_i - P_i(Z_i)) ≥ Ω
            %
            %   where:
            %   - Y is the dual variable for the TP constraint
            %   - Z_i are dual variables for symmetry constraints
            %   - P_i permutes output systems i and i+1
            %
            % DUAL VARIABLES:
            %   Y     - 2×2 hermitian matrix (TP constraint)
            %   Z_i   - dim_total × dim_total hermitian matrices (symmetry)
            %           One for each adjacent transposition (N-1 total)
            %
            % STRONG DUALITY:
            %   Under Slater's condition (which holds here), strong duality
            %   guarantees that the primal and dual optimal values are equal:
            %   
            %   max_E Tr(E*Ω) = min_Y Tr(Y)
            %
            %   This provides a certificate of optimality.
            %
            % USAGE:
            %   The dual problem can be useful for:
            %   - Verifying optimality (primal = dual)
            %   - Obtaining dual certificates
            %   - Theoretical analysis
            %
            % SEE ALSO:
            %   solvePrimal
            
            dim_total = 2^(obj.N+1);
            dim_outputs = 2^obj.N;
            dims = 2 * ones(1, obj.N+1);
            
            cvx_begin sdp quiet
                variable Y(2, 2) hermitian
                
                % Dual variables for each adjacent transposition constraint
                % We need (N-1) dual variables for symmetry constraints
                variable Z_all(dim_total, dim_total, obj.N-1) hermitian
                
                minimize( real(trace(Y)) )
                
                subject to
                    % Construct the dual constraint
                    % Start with TP term
                    dual_matrix = kron(Y, eye(dim_outputs));
                    
                    % Add symmetry constraint terms
                    for i = 1:(obj.N-1)
                        Z = Z_all(:,:,i);
                        
                        % Permutation for swapping outputs i and i+1
                        perm = 1:(obj.N+1);
                        perm(i+1) = i+2;
                        perm(i+2) = i+1;
                        
                        % Add antisymmetric part: Z - P(Z)
                        dual_matrix = dual_matrix + ...
                            (Z - PermuteSystems(Z, perm, dims));
                    end
                    
                    % Dual constraint
                    dual_matrix >= obj.Omega;
            cvx_end
            
            optimum = cvx_optval;
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
            %               Eigenvalues below threshold are discarded
            %
            % OUTPUT:
            %   kraus_ops    - Cell array of Kraus operators {K_1, K_2, ..., K_r}
            %                  Each K_i is (2^N × 2) mapping input to N outputs
            %   eigenvalues  - All eigenvalues of E in descending order
            %
            % DESCRIPTION:
            %   Computes Kraus operators via eigenvalue decomposition of the
            %   Choi matrix. The Choi-Jamiolkowski isomorphism relates the
            %   Choi matrix to Kraus operators:
            %
            %   E = Σ_i |K_i⟩⟩⟨⟨K_i|
            %
            %   where |K_i⟩⟩ is the vectorization of K_i.
            %
            % ALGORITHM:
            %   1. Symmetrize E to ensure exact hermiticity
            %   2. Compute eigendecomposition: E = V D V†
            %   3. Filter eigenvalues above threshold
            %   4. For each eigenvalue λ_i with eigenvector v_i:
            %      K_i = √λ_i * reshape(v_i)
            %
            % CHOI-JAMIOLKOWSKI CONVENTION:
            %   This implementation uses the Input* ⊗ Output convention:
            %   E_{(i⊗j),(k⊗l)} = K_{j,l} * conj(K_{i,k})
            %
            % RANK AND REPRESENTATION:
            %   - Minimal Kraus rank = number of positive eigenvalues
            %   - Physical channels: rank(E) ≤ (dim_in × dim_out)²
            %   - For optimal cloning: typically low rank due to structure
            %
            % THRESHOLD SELECTION:
            %   - Too small (e.g., 1e-15): May include numerical noise
            %   - Too large (e.g., 1e-6): May discard physical operators
            %   - Default 1e-10: Good balance for double precision
            %
            % SEE ALSO:
            %   verifyKrausOperators, applyChannel
            
            arguments
                obj
                E {mustBeNumeric}
                threshold = 1e-10
            end
            
            dim_in = 2;           % Input dimension
            dim_out = 2^obj.N;    % Output dimension (N qubits)
            dim_total = dim_in * dim_out;
            
            % Verify matrix dimensions
            if ~isequal(size(E), [dim_total, dim_total])
                error('Choi matrix E has incorrect dimensions. Expected %dx%d, got %dx%d', ...
                      dim_total, dim_total, size(E,1), size(E,2));
            end
            
            % Ensure E is Hermitian (should be from optimization)
            E = (E + E') / 2;
            
            % Eigenvalue decomposition of the Choi matrix
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
                % 1. Get eigenvector
                vec_K = V_filtered(:, i);
                
                % 2. Reshape for (in \otimes out) convention:
                K = reshape(vec_K, [dim_out, dim_in]);
                
                % 3. Scale with sqrt of the eigenvalue
                K = sqrt(eigenvalues_filtered(i)) * K;
                
                kraus_ops{i} = K;
            end
            
            % Return all eigenvalues (including filtered ones) for analysis
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
            %   kraus_ops - Cell array of Kraus operators {K_1, ..., K_r}
            %   tol       - (optional) Tolerance for verification
            %               Default: 1e-6
            %
            % OUTPUT:
            %   is_valid           - Boolean: true if completeness satisfied
            %   completeness_error - Frobenius norm ||Σ_i K_i† K_i - I||_F
            %
            % DESCRIPTION:
            %   Verifies that the Kraus operators satisfy the completeness
            %   relation, which is equivalent to the trace-preserving (TP)
            %   condition for quantum channels:
            %
            %   Σ_i K_i† K_i = I_{dim_in}
            %
            %   This ensures that the channel preserves the trace of density
            %   matrices: Tr(ρ_out) = Tr(ρ_in) for all ρ_in.
            %
            % MATHEMATICAL BACKGROUND:
            %   A completely positive trace-preserving (CPTP) map has a
            %   Kraus representation:
            %
            %   Φ(ρ) = Σ_i K_i ρ K_i†
            %
            %   The TP condition Tr(Φ(ρ)) = Tr(ρ) is equivalent to the
            %   completeness relation above.
            %
            % INTERPRETATION:
            %   completeness_error < tol:  Channel is properly normalized
            %   completeness_error ≥ tol:  Numerical issues or incorrect
            %                              Kraus decomposition
            %
            % TYPICAL VALUES:
            %   < 1e-10: Excellent (near machine precision)
            %   < 1e-6:  Good (default tolerance)
            %   < 1e-4:  Acceptable (may indicate numerical issues)
            %   ≥ 1e-4:  Poor (likely incorrect decomposition)
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
            
            dim_in = 2;  % Input dimension
            
            % Compute sum_i K_i† K_i (should equal identity)
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
            % applyChannel - Apply quantum channel to an input state
            %
            % SYNTAX:
            %   rho_out = applyChannel(obj, kraus_ops, rho_in)
            %
            % INPUT:
            %   kraus_ops - Cell array of Kraus operators {K_1, ..., K_r}
            %   rho_in    - Input density matrix (2 × 2 hermitian, Tr=1)
            %
            % OUTPUT:
            %   rho_out   - Output density matrix (2^N × 2^N hermitian, Tr=1)
            %
            % DESCRIPTION:
            %   Applies the quantum channel defined by the Kraus operators
            %   to the input state:
            %
            %   ρ_out = Φ(ρ_in) = Σ_i K_i ρ_in K_i†
            %
            %   This implements the action of the 1->N cloning map on a
            %   single input qubit, producing N output qubits in a joint
            %   state ρ_out.
            %
            % PROPERTIES OF OUTPUT:
            %   - Hermitian: ρ_out = ρ_out†
            %   - Positive semidefinite: ρ_out ≥ 0
            %   - Normalized: Tr(ρ_out) = 1 (if Kraus ops satisfy completeness)
            %   - Dimension: 2^N × 2^N (N qubits)
            %
            % QUANTUM CLONING INTERPRETATION:
            %   The output state ρ_out lives in the Hilbert space of N qubits.
            %   To analyze individual clones, compute reduced density matrices:
            %   
            %   ρ_1 = Tr_{2,...,N}(ρ_out)  (first clone)
            %   ρ_2 = Tr_{1,3,...,N}(ρ_out)  (second clone)
            %
            %   Due to symmetry: ρ_1 = ρ_2 = ... = ρ_N
            %
            % FIDELITY COMPUTATION:
            %   To compute cloning fidelity for state |ψ⟩:
            %   
            %   rho_in = |ψ⟩⟨ψ|
            %   rho_out = applyChannel(kraus_ops, rho_in)
            %   rho_clone = PartialTrace(rho_out, [2:N], dims)
            %   fidelity = Tr(rho_clone * rho_in)
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
            %   E   - Choi matrix to verify (dim_total × dim_total hermitian)
            %   tol - (optional) Tolerance for constraint satisfaction
            %         Default: 1e-6
            %
            % OUTPUT:
            %   is_valid   - Boolean: true if all constraints satisfied
            %   violations - Structure with fields:
            %                .CP       - Minimum eigenvalue of E (should be ≥ 0)
            %                .TP       - ||Tr_out(E) - I||_F (should be ≈ 0)
            %                .symmetry - Array of N-1 symmetry violations
            %
            % DESCRIPTION:
            %   Verifies that a given Choi matrix E satisfies all three
            %   constraints of the primal optimization problem:
            %
            %   1. Complete Positivity (CP): E ≥ 0
            %      Checked via minimum eigenvalue
            %
            %   2. Trace Preserving (TP): Tr_out(E) = I
            %      Checked via Frobenius norm of difference
            %
            %   3. Permutation Symmetry: P_i(E) = E for all i
            %      Checked for all N-1 adjacent transpositions
            %
            % CONSTRAINT DETAILS:
            %
            %   CP Constraint:
            %   - violations.CP = min(eig(E))
            %   - Satisfied if: violations.CP ≥ -tol
            %   - Typical values: 0 to 1e-10 (good), -1e-12 (acceptable),
            %                     < -1e-6 (violation)
            %
            %   TP Constraint:
            %   - violations.TP = ||Tr_{out}(E) - I_2||_F
            %   - Satisfied if: violations.TP ≤ tol
            %   - Typical values: < 1e-10 (excellent), < 1e-6 (good),
            %                     > 1e-4 (violation)
            %
            %   Symmetry Constraints:
            %   - violations.symmetry(i) = ||E - P_{i,i+1}(E)||_F
            %   - Satisfied if: all(violations.symmetry ≤ tol)
            %   - For i = 1,...,N-1 (adjacent transpositions)
            %
            % OVERALL VALIDITY:
            %   is_valid = true if and only if:
            %   - violations.CP ≥ -tol  AND
            %   - violations.TP ≤ tol   AND
            %   - all(violations.symmetry ≤ tol)
            %
            % INTERPRETATION GUIDE:
            %   violations.CP:
            %     ≥ 0:        Perfect (matrix is PSD)
            %     ≥ -1e-10:   Excellent (numerical zero)
            %     ≥ -1e-6:    Acceptable (small numerical error)
            %     < -1e-6:    Violation (matrix not PSD)
            %
            %   violations.TP:
            %     < 1e-14:    Perfect (machine precision)
            %     < 1e-10:    Excellent
            %     < 1e-6:     Good
            %     < 1e-4:     Acceptable
            %     ≥ 1e-4:     Violation
            %
            %   violations.symmetry:
            %     All = 0:    Exact symmetry (exceptional!)
            %     All < 1e-10: Excellent
            %     All < 1e-6:  Good
            %     Any ≥ 1e-6:  Violation
            %
            % USAGE SCENARIOS:
            %   1. After solving SDP: Verify numerical quality
            %   2. Debugging: Identify which constraints are problematic
            %   3. Solver comparison: Compare solution quality across solvers
            %   4. Publication: Report constraint satisfaction for validation
            %
            % EXAMPLE 1: Basic verification
            %   [fid, E] = cloner.solvePrimal();
            %   [valid, viol] = cloner.verifyPrimal(E);
            %   if valid
            %       fprintf('✓ Solution is valid\n');
            %   else
            %       fprintf('✗ Constraint violations detected\n');
            %   end
            %
            % SEE ALSO:
            %   solvePrimal, verifyKrausOperators
            
            arguments
                obj
                E {mustBeNumeric}
                tol = 1e-6
            end
            
            violations = struct();
            dims = 2 * ones(1, obj.N+1);
            
            % Check CP
            eigs_E = eig(E);
            violations.CP = min(eigs_E);
            
            % Check TP
            TP_result = PartialTrace(E, 2:(obj.N+1), dims);
            violations.TP = norm(TP_result - eye(2), 'fro');
            
            % Check symmetry for all adjacent swaps
            violations.symmetry = zeros(obj.N-1, 1);
            for i = 2:obj.N
                perm = 1:(obj.N+1);
                perm(i) = i+1;
                perm(i+1) = i;
                
                E_perm = PermuteSystems(E, perm, dims);
                violations.symmetry(i-1) = norm(E - E_perm, 'fro');
            end
            
            % Overall validity check
            is_valid = violations.CP >= -tol && ...
                       violations.TP <= tol && ...
                       all(violations.symmetry <= tol);
        end
        
    end
end