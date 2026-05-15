classdef SDP_Symmetric_1to3_Cloner
    properties
        Omega
    end

    methods
        % Constructor
        function obj = SDP_Symmetric_1to3_Cloner(states)
            arguments
                states;
            end

            tmpOmega = zeros(16, 16);
            for state = states
                rho = state * state'; % Density matrix for pure state

                part_in  = conj(rho);
                part_out = kron(rho, eye(4) );
                tmpOmega = tmpOmega + kron( part_in, part_out );
            end

            % Normalize
            obj.Omega = tmpOmega / length(states);
        end

        % Solve the primal SDP problem
        function [primal__optimum, E] = solvePrimal(obj)

            cvx_begin sdp quiet
                variable E(16, 16) hermitian

                % Objective: maximize entanglement fidelity
                maximize( real(trace(E * obj.Omega)) )

                subject to
                    % 1. Complete positivity
                    E >= 0;

                    % 2. Trace-preserving condition
                    PartialTrace(E, [2, 3, 4], [2, 2, 2, 2]) == eye(2);

                    % 3. Symmetry: swap of output subsystems
                    dims   = [2, 2, 2, 2];
                    perm1  = [1, 3, 2, 4];
                    perm2  = [1, 4, 3, 2];
                    E == PermuteSystems(E, perm1, dims);
                    E == PermuteSystems(E, perm2, dims);
            cvx_end

            primal__optimum = cvx_optval;
        end

        function [dual_optimum, Y] = solveDual(obj)
            
            cvx_begin sdp quiet
                variable Y(2, 2) hermitian
                variable Z1(16, 16) hermitian 
                variable Z2(16, 16) hermitian 
                
                minimize( real(trace(Y)) )
                
                subject to
                    dims   = [2, 2, 2, 2];
                    perm1  = [1, 3, 2, 4];
                    perm2  = [1, 4, 3, 2];
                    C1 = PermuteSystems(Z1, perm1, dims);
                    C2 = PermuteSystems(Z2, perm2, dims);

                    % Dual constraint
                    kron(Y, eye(8)) + (Z1 - C1) + (Z2 - C2) >= obj.Omega;
            cvx_end
            
            dual_optimum = cvx_optval;
        end
    end
end