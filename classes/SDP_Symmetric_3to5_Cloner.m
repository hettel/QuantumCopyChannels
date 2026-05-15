classdef SDP_Symmetric_3to5_Cloner
    properties
        Omega
    end

    methods
        % Constructor
        function obj = SDP_Symmetric_3to5_Cloner(states)
            arguments
                states;
            end

            tmpOmega = zeros(256, 256);
            for state = states
                rho = state * state'; % Density matrix for pure state

                part_in  = Tensor(conj(rho), 3);
                part_out = kron(rho, eye(16) );
                tmpOmega = tmpOmega + kron( part_in, part_out );
            end

            % Normalize
            obj.Omega = tmpOmega / length(states);
        end

        % Solve the primal SDP problem
        function [primal__optimum, E] = solvePrimal(obj)

            cvx_begin sdp quiet
                variable E(256, 256) hermitian

                % Objective: maximize entanglement fidelity
                maximize( real(trace(E * obj.Omega)) )

                subject to
                    % 1. Complete positivity
                    E >= 0;

                    % 2. Trace-preserving condition
                    PartialTrace(E, [4, 5, 6, 7, 8], [2, 2, 2, 2, 2, 2 ,2, 2]) == eye(8);

                    % 3. Symmetry: swap of output subsystems
                    dims  = [2, 2, 2, 2, 2, 2, 2, 2];
                    perm1 = [1, 2, 3, 5, 4, 6, 7, 8];
                    perm2 = [1, 2, 3, 6, 5, 4, 7, 8];
                    perm3 = [1, 2, 3, 7, 5, 6, 4, 8];
                    perm4 = [1, 2, 3, 8, 5, 6, 7, 4];
                   
                    E == PermuteSystems(E, perm1, dims);
                    E == PermuteSystems(E, perm2, dims);
                    E == PermuteSystems(E, perm3, dims);
                    E == PermuteSystems(E, perm4, dims);
            cvx_end

            primal__optimum = cvx_optval;
        end

        function [dual_optimum, Y] = solveDual(obj)
            
            cvx_begin sdp quiet
                variable Y(8, 8) hermitian
                variable Z1(256, 256) hermitian 
                variable Z2(256, 256) hermitian 
                variable Z3(256, 256) hermitian 
                variable Z4(256, 256) hermitian 
                
                minimize( real(trace(Y)) )
                
                subject to
                    dims  = [2, 2, 2, 2, 2, 2, 2, 2];
                    perm1 = [1, 2, 3, 5, 4, 6, 7, 8];
                    perm2 = [1, 2, 3, 6, 5, 4, 7, 8];
                    perm3 = [1, 2, 3, 7, 5, 6, 4, 8];
                    perm4 = [1, 2, 3, 8, 5, 6, 7, 4];
                   
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