classdef SDP_Symmetric_2to5_Cloner
    properties
        Omega
    end

    methods
        % Constructor
        function obj = SDP_Symmetric_2to5_Cloner(states)
            arguments
                states;
            end

            tmpOmega = zeros(128, 128);
            for state = states
                rho = state * state'; % Density matrix for pure state

                part_in  = Tensor(conj(rho),2);
                part_out = kron(rho, eye(16) );
                tmpOmega = tmpOmega + kron( part_in, part_out );
            end

            % Normalize
            obj.Omega = tmpOmega / length(states);
        end

        % Solve the primal SDP problem
        function [primal__optimum, E] = solvePrimal(obj)

            cvx_begin sdp quiet
                variable E(128, 128) hermitian

                % Objective: maximize entanglement fidelity
                maximize( real(trace(E * obj.Omega)) )

                subject to
                    % 1. Complete positivity
                    E >= 0;

                    % 2. Trace-preserving condition
                    PartialTrace(E, [3, 4, 5, 6, 7], [2, 2, 2, 2, 2, 2, 2]) == eye(4);

                    % 3. Symmetry: swap of output subsystems
                    dims  = [2, 2, 2, 2, 2, 2, 2];
                    perm1 = [1, 2, 4, 3, 5, 6, 7];
                    perm2 = [1, 2, 5, 4, 3, 6, 7];
                    perm3 = [1, 2, 6, 4, 5, 3, 7];
                    perm4 = [1, 2, 7, 4, 5, 6, 3];

                    E == PermuteSystems(E, perm1, dims);
                    E == PermuteSystems(E, perm2, dims);
                    E == PermuteSystems(E, perm3, dims);
                    E == PermuteSystems(E, perm4, dims);
            cvx_end

            primal__optimum = cvx_optval;
        end

        function [dual_optimum, Y] = solveDual(obj)
            
            cvx_begin sdp quiet
                variable Y(4, 4) hermitian
                variable Z1(128, 128) hermitian 
                variable Z2(128, 128) hermitian 
                variable Z3(128, 128) hermitian 
                variable Z4(128, 128) hermitian 
                
                minimize( real(trace(Y)) )
                
                subject to
                    dims  = [2, 2, 2, 2, 2, 2, 2];
                    perm1 = [1, 2, 4, 3, 5, 6, 7];
                    perm2 = [1, 2, 5, 4, 3, 6, 7];
                    perm3 = [1, 2, 6, 4, 5, 3, 7];
                    perm4 = [1, 2, 7, 4, 5, 6, 3];
                    C1 = PermuteSystems(Z1, perm1, dims);
                    C2 = PermuteSystems(Z2, perm2, dims);
                    C3 = PermuteSystems(Z3, perm3, dims);
                    C4 = PermuteSystems(Z4, perm4, dims);
                    % Dual constraint
                    kron(Y, eye(32)) + (Z1 - C1) + (Z2 - C2) + (Z3 - C3) + (Z4 - C4) >= obj.Omega;
            cvx_end
            
            dual_optimum = cvx_optval;
        end
    end
end