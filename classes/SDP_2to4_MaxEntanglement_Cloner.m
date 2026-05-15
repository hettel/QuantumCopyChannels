classdef SDP_2to4_MaxEntanglement_Cloner
    properties
        Omega
    end

    methods
        % Constructor
        function obj = SDP_2to4_MaxEntanglement_Cloner(states)
            arguments
                states;
            end

            tmpOmega = zeros(64, 64);
            for state = states
                rho = state * state'; % Pure state density matrix
                tmpOmega = tmpOmega + kron(kron(conj(rho), rho), eye(4));
            end

            obj.Omega = tmpOmega / length(states);
        end

        % Solving the primal SDP problem
        function [optimum, E] = solvePrimal(obj)
 
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

