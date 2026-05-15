classdef SDP_2to4_MaxEntanglement_RealAmpl_Cloner
    % SDP_2to4_MaxEntanglement_RealAmpl_Cloner
    %
    %   This class constructs and solves a semidefinite program (SDP) for
    %   optimal 2→4 cloning of maximally entangled states with real
    %   amplitudes. It formulates both the primal and dual SDP problems
    %   used to compute the optimal fidelity of the cloning channel.
    %
    %   The channel is represented by a Choi matrix E \in C^{64×64}, and
    %   symmetry + trace-preservation constraints are enforced via CVX.

    properties
        % Omega – averaged operator constructed from the given ensemble
        %         of input states. It appears in the objective function
        %         for maximizing the cloning fidelity.
        Omega
    end

    methods
        % Constructor
        function obj = SDP_2to4_MaxEntanglement_RealAmpl_Cloner()

            % Define a symmetric set of input states (real amplitudes).
            % These 4-dimensional vectors represent 2-qubit pure states.
            % The ensemble is used for computing the averaged operator Ω.

            s1 = [1; 0; 0; 1]/sqrt(2);
            s2 = [1; 0; 0; -1]/sqrt(2);
            s3 = [0; 1; 1; 0]/sqrt(2);
            s4 = [0; 1; -1; 0]/sqrt(2);

            % Four additional symmetric real states
            s5 = [-1; 1; 1; 1]/2;
            s6 = [1; -1; 1; 1]/2;
            s7 = [1; 1; -1; 1]/2;
            s8 = [1; 1; 1; -1]/2;

            % Collect all states into one matrix (each column is a state vector)
            states = [s1, s2, s3, s4, s5, s6, s7, s8];

            % Build Ω = (1/N) * Σ_i kron( kron(ρ_i*, ρ_i), I_4 )
            % where ρ_i = |ψ_i⟩⟨ψ_i|.
            %
            % The structure corresponds to the Choi-matrix objective for
            % symmetric cloning maps and is typical in entanglement-fidelity
            % optimization.

            tmpOmega = zeros(64, 64);
            for state = states
                rho = state * state'; % Pure state density matrix
                tmpOmega = tmpOmega + kron(kron(conj(rho), rho), eye(4));
            end

            % Normalize by number of states
            obj.Omega = tmpOmega / length(states);
        end

        % Solving the primal SDP problem
        function [optimum, E] = solvePrimal(obj)
            % The primal problem maximizes the average fidelity:
            %
            %       maximize Tr(E * Ω)
            %
            % under the constraints that:
            %   (1) E >= 0                      (complete positivity)
            %   (2) Tr_out(E) = I               (trace preservation)
            %   (3) E = Permute(E)              (symmetry constraints)

            cvx_begin sdp quiet
                variable E(64, 64) hermitian

                % Objective: maximize fidelity with respect to Omega
                maximize( real(trace(E * obj.Omega)) )

                subject to
                    % 1. Completely positive channel (Choi matrix must be PSD)
                    E >= 0;

                    % 2. Trace-preserving condition:
                    % Partial trace over systems 2 and 3 yields I_4
                    PartialTrace(E, [2, 3], [4, 4, 4]) == eye(4);

                    % 3. Symmetry constraint under exchange of subsystems
                    dims = [4, 4, 4];
                    perm = [1, 3, 2];
                    E == PermuteSystems(E, perm, dims);
            cvx_end

            optimum = cvx_optval;
        end

        % Solving the dual SDP problem
        function [optimum, Y] = solveDual(obj)
            % The dual program is derived from the Lagrangian of the primal SDP.
            % It provides a lower bound on the optimal fidelity.
            %
            % Dual variables:
            %   Y  – corresponds to the trace-preserving constraint
            %   Z  – corresponds to the symmetry constraint
            %
            % The dual objective is:
            %       minimize Tr(Y)
            %
            % subject to:
            %   kron(Y, I) + (Z - Permute(Z)) >= Omega

            cvx_begin sdp quiet
                variable Y(4, 4) hermitian
                variable Z(64, 64) hermitian

                % Objective: minimize trace of Y
                minimize( real(trace(Y)) )

                subject to
                    dims = [4, 4, 4];
                    perm = [1, 3, 2];

                    % Main dual constraint
                    kron(Y, eye(16)) + (Z - PermuteSystems(Z, perm, dims)) ...
                        >= obj.Omega;
            cvx_end

            optimum = cvx_optval;
        end
    end
end

