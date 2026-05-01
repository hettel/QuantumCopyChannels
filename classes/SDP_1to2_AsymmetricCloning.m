classdef SDP_1to2_AsymmetricCloning
    % Detailed explanation goes here

    properties
        States
        lambda
        Omega_A
        Omega_B
    end

    methods
        % Contructor
        function obj = SDP_1to2_AsymmetricCloning(states, lambda)
            arguments
                states {mustBeNumeric} 
                lambda {mustBeNumeric} 
            end

            obj.States = states;
            obj.lambda = lambda;

            tmpOmega_A = zeros(8, 8);
            tmpOmega_B = zeros(8, 8);
            for state = states
                rho = state * state';
                
                tmpOmega_A = tmpOmega_A + kron( kron(conj(rho), rho), eye(2) );
                tmpOmega_B = tmpOmega_B + kron( kron(conj(rho), eye(2)), rho );
            end          
            obj.Omega_A = tmpOmega_A / length(states);
            obj.Omega_B = tmpOmega_B / length(states);
        end

        % Soling the primary problem
        function [optimum, E] = solvePrimal(obj)
    
            cvx_begin sdp quiet
                variable E(8, 8) hermitian
                
                % Objective function: Max fidelity
                maximize( real( obj.lambda*trace(E * obj.Omega_A) + (1-obj.lambda)*trace(E * obj.Omega_B) ) )
                
                subject to
                    % 1. Completely Positive (CP)
                    E >= 0;
                    
                    % 2. Trace Preserving (TP): Tr_{ABC}(E) = I_in
                    PartialTrace(E, [2, 3], [2, 2, 2]) == eye(2);
            cvx_end

            optimum = cvx_optval;
        end

        % Soving the dual problem
        function [optimum, Y] = solveDual(obj)
            
            cvx_begin sdp quiet
                variable Y(2, 2) hermitian
                variable Z(8, 8) hermitian % Dual-Variable für A <-> B
                 
                minimize( real(trace(Y)) )
                
                subject to
                    % TP-Bedingung
                    kron(Y, eye(4)) >= (obj.lambda * obj.Omega_A + (1-obj.lambda) * obj.Omega_B);
            cvx_end
    
            optimum = cvx_optval;
        end
    end
end