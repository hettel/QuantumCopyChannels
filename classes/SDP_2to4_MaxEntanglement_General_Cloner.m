classdef SDP_2to4_MaxEntanglement_General_Cloner
    % SDP_2to4_MaxEntanglement_General_Cloner
    %
    %   This class constructs and solves a semidefinite program (SDP) for
    %   optimal 2→4 cloning of maximally entangled states. In contrast to
    %   the "RealAmpl" variant, this class uses a *general* ensemble of
    %   maximally entangled bipartite states, sampled uniformly using a
    %   high-quality Fibonacci lattice (quasi–Monte Carlo).
    %
    %   The ensemble is used to build the averaged operator Ω, which appears
    %   in the primal and dual SDP problems.
    %
    %   Requires:
    %       - CVX (for SDP solving)
    %       - generate_mbe_state() to generate maximally entangled states
    %         parameterized by 7 angles.

    properties
        % Omega – averaged operator constructed from a large sampled
        %         ensemble of maximally entangled states
        Omega
    end

    methods
        % Constructor
        function obj = SDP_2to4_MaxEntanglement_General_Cloner(n_states)
            % Constructor argument:
            %   n_states – number of maximally entangled states to sample.
            %              These are drawn using a Fibonacci sphere grid to
            %              ensure quasi-uniform coverage of the state space.
            %
            % Default: 200000 states (large but efficient for QMC sampling).

            arguments
                n_states (1,1) {mustBeInteger, mustBePositive} = 200000;
            end

            % Preallocate cell array of pure states
            states_fibonacci = cell(n_states, 1);

            % Golden ratio – used for generating the Fibonacci lattice
            % to achieve quasi-uniform sampling on a multidimensional torus.
            phi_golden = (1 + sqrt(5)) / 2;

            % Generate n_states maximally entangled states using
            % different quasi-random angle parameters.
            for k = 1:n_states

                % Map k to [0,1] for continuous angular coverage
                t = k / n_states;

                % Use Fibonacci-based angle sequences to approximate
                % uniform sampling across several angular dimensions.

                % First subsystem angles
                theta1 = acos(1 - 2*t);
                phi1   = 2 * pi * k / phi_golden;
                psi1   = 2 * pi * mod(k * phi_golden, 1);

                % Second subsystem angles
                theta2 = acos(1 - 2*mod(k/2, 1));
                phi2   = 2 * pi * mod(k / phi_golden^2, 1);
                psi2   = pi * mod(k * sqrt(2), 1);

                % Global phase-like entanglement parameter
                gamma  = 2 * pi * mod(k / sqrt(3), 1);

                % Construct a maximally entangled bipartite state
                % parameterized by 7 angles.
                states_fibonacci{k} = generate_mbe_state( ...
                    theta1, phi1, psi1, theta2, phi2, psi2, gamma );
            end

            % Convert cell array to a matrix of state vectors
            states = [ states_fibonacci{:} ];

            % Build the averaged operator Ω
            %   Omega = (1/N) * Σ_i kron( kron(rho_i*, rho_i), I_4 )
            %
            % where rho_i = |ψ_i⟩⟨ψ_i|.

            tmpOmega = zeros(64, 64);
            for state = states
                rho = state * state'; % Density matrix for pure state

                tmpOmega = tmpOmega + kron( kron(conj(rho), rho), eye(4) );
            end

            % Normalize
            obj.Omega = tmpOmega / length(states);
        end

        % Solve the primal SDP problem
        function [optimum, E] = solvePrimal(obj)
            % The primal problem:
            %
            %   maximize   Tr(E * Omega)
            %
            % subject to:
            %   1. E >= 0                      (complete positivity)
            %   2. Tr_out(E) = I               (trace preservation)
            %   3. E = Permute(E)              (symmetry constraints)
            %
            % E is the Choi matrix of the 2→4 cloning channel.

            cvx_begin sdp quiet
                variable E(64, 64) hermitian

                % Objective: maximize entanglement fidelity
                maximize( real(trace(E * obj.Omega)) )

                subject to
                    % 1. Complete positivity
                    E >= 0;

                    % 2. Trace-preserving condition
                    PartialTrace(E, [2, 3], [4, 4, 4]) == eye(4);

                    % 3. Symmetry: swap of output subsystems
                    dims = [4, 4, 4];
                    perm = [1, 3, 2];
                    E == PermuteSystems(E, perm, dims);
            cvx_end

            optimum = cvx_optval;
        end

        % Solve the dual SDP problem
        function [optimum, Y] = solveDual(obj)
            % The dual of the primal SDP:
            %
            %   minimize Tr(Y)
            %
            % subject to:
            %   kron(Y, I_16) + (Z - Permute(Z)) >= Omega
            %
            % Dual variables correspond to the primal constraints:
            %   Y – trace-preserving constraint
            %   Z – symmetry constraint

            cvx_begin sdp quiet
                variable Y(4, 4) hermitian
                variable Z(64, 64) hermitian

                % Objective: minimize trace of Y
                minimize( real(trace(Y)) )

                subject to
                    dims = [4, 4, 4];
                    perm = [1, 3, 2];

                    kron(Y, eye(16)) + (Z - PermuteSystems(Z, perm, dims)) ...
                        >= obj.Omega;
            cvx_end

            optimum = cvx_optval;
        end
    end
end


%% Helper function: Generate maximally entangled bipartite state
function psi = generate_mbe_state(theta1, phi1, psi1, theta2, phi2, psi2, gamma)
    % generate_mbe_state
    %
    %   Generates a maximally entangled bipartite pure state |ψ⟩ in C^2 ⊗ C^2.
    %   The state is obtained by applying local SU(2) unitaries U_A ⊗ U_B
    %   to a phase-modified Bell state |Φ+⟩.
    %
    % INPUT:
    %   theta1, phi1, psi1 : Euler angles (ZYZ convention) for local unitary U_A
    %   theta2, phi2, psi2 : Euler angles (ZYZ convention) for local unitary U_B
    %   gamma              : additional global phase applied to |Φ+⟩
    %
    % PURPOSE:
    %   Any maximally entangled 2-qubit pure state can be written as
    %
    %       |ψ⟩ = (U_A ⊗ U_B) |Φ+⟩ ,
    %
    %   where U_A, U_B ∈ SU(2). This function generates such a state using
    %   arbitrary Euler-angle parameterizations for U_A and U_B.
    %
    %   Used for sampling random maximally entangled states in the SDP
    %   optimization of quantum cloning channels.

    % Construct local unitary operators (SU(2), ZYZ parameterization)
    U_A = euler_to_unitary(theta1, phi1, psi1);
    U_B = euler_to_unitary(theta2, phi2, psi2);

    % Reference Bell state |Φ+⟩ = (|00⟩ + |11⟩) / sqrt(2)
    % with an additional adjustable phase e^{i γ} on |11⟩.
    bell = [1; 0; 0; exp(1i * gamma)] / sqrt(2);

    % Apply local unitaries: |ψ⟩ = (U_A ⊗ U_B) |Φ+⟩
    U = kron(U_A, U_B);
    psi = U * bell;

    % Normalize result (normally unnecessary but ensures numerical stability)
    psi = psi / norm(psi);
end


%% Convert Euler angles (ZYZ) to SU(2) unitary
function U = euler_to_unitary(theta, phi, psi)
    % euler_to_unitary
    %
    %   Converts Euler angles (θ, φ, ψ) in the ZYZ decomposition into a
    %   2×2 SU(2) rotation matrix:
    %
    %       U = R_z(ψ) * R_y(θ) * R_z(φ)
    %
    %   where:
    %       R_z(α) = exp(-i α σ_z / 2)
    %       R_y(β) = exp(-i β σ_y / 2)
    %
    % INPUT:
    %   theta : rotation about Y axis
    %   phi   : first rotation about Z axis
    %   psi   : second rotation about Z axis
    %
    % PURPOSE:
    %   This parameterization spans all elements of SU(2) and is widely used
    %   in quantum information for generating arbitrary single-qubit unitaries.

    % Rotation around Z by φ
    Rz_phi = [exp(-1i * phi/2), 0;
              0,                exp(1i * phi/2)];

    % Rotation around Y by θ
    Ry_theta = [cos(theta/2), -sin(theta/2);
                sin(theta/2),  cos(theta/2)];

    % Rotation around Z by ψ
    Rz_psi = [exp(-1i * psi/2), 0;
              0,                exp(1i * psi/2)];

    % Combine rotations (ZYZ convention)
    U = Rz_psi * Ry_theta * Rz_phi;
end