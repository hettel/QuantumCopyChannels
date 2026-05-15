classdef SDP_Symmetric_1to2_Cloner
    properties
        Omega
    end

    methods
        % Constructor
        function obj = SDP_Symmetric_1to2_Cloner(states)
            arguments
                states;
            end

            tmpOmega = zeros(8, 8);
            for state = states
                rho = state * state'; % Density matrix for pure state

                part_in  = conj(rho);
                part_out = kron(rho, eye(2) );
                tmpOmega = tmpOmega + kron( part_in, part_out );
            end

            % Normalize
            obj.Omega = tmpOmega / length(states);
        end

        % Solve the primal SDP problem
        function [primal__optimum, E] = solvePrimal(obj)

            cvx_begin sdp quiet
                variable E(8, 8) hermitian

                % Objective: maximize entanglement fidelity
                maximize( real(trace(E * obj.Omega)) )

                subject to
                    % 1. Complete positivity
                    E >= 0;

                    % 2. Trace-preserving condition
                    PartialTrace(E, [2, 3], [2, 2, 2]) == eye(2);

                    % 3. Symmetry: swap of output subsystems
                    dims  = [2, 2, 2];
                    perm  = [1, 3, 2];
                    E == PermuteSystems(E, perm, dims);
            cvx_end

            primal__optimum = cvx_optval;
        end

        function [dual_optimum, Y] = solveDual(obj)
            
            cvx_begin sdp quiet
                variable Y(2, 2) hermitian
                variable Z(8, 8) hermitian 
                
                minimize( real(trace(Y)) )
                
                subject to
                    dims  = [2, 2, 2];
                    perm  = [1, 3, 2];
                    C = PermuteSystems(Z, perm, dims);

                    % Dual constraint
                    kron(Y, eye(4)) + (Z - C) >= obj.Omega;
            cvx_end
            
            dual_optimum = cvx_optval;
        end
    end
end