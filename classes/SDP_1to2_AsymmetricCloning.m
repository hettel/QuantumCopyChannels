classdef SDP_1to2_AsymmetricCloning
    % ================================================================
    %  SDP_1to2_AsymmetricCloning
    %
    %  This class implements the semidefinite programs (SDPs) for
    %  asymmetric 1→2 qubit cloning using the
    %  Jamiołkowski–Choi representation of quantum channels.
    %
    %  A cloning map
    %       Φ : 2  →  (2 ⊗ 2)
    %
    %  is represented by its Choi operator E, an 8×8 matrix.
    %
    %  Objective:
    %     Maximize  F = λ F_A + (1 − λ) F_B
    %
    %  where λ ∈ [0,1] controls the asymmetry between clone A and B.
    %
    %  Ω_A and Ω_B are precomputed operators representing the fidelity
    %  functionals for output A and output B.
    %
    % ================================================================

    properties
        States      % Input pure states |ψ>, stored as 2×N matrix
        lambda      % Asymmetry parameter λ ∈ [0,1]
        Omega_A     % Fidelity operator for output clone A
        Omega_B     % Fidelity operator for output clone B
    end

    methods
        %% ------------------------------------------------------------
        %  Constructor
        %
        %  Computes the two averaging operators Ω_A and Ω_B via:
        %
        %    Ω_A = (1/N) Σ  conj(ρ) ⊗ ρ ⊗ I
        %    Ω_B = (1/N) Σ  conj(ρ) ⊗ I ⊗ ρ
        %
        %  where ρ = |ψ><ψ|.
        %
        %  These operators encode the fidelity functional for clone A
        %  and clone B inside the SDP objective.
        % ------------------------------------------------------------
        function obj = SDP_1to2_AsymmetricCloning(states, lambda)
            arguments
                states {mustBeNumeric}
                lambda {mustBeNumeric}
            end

            obj.States = states;
            obj.lambda = lambda;

            tmpA = zeros(8, 8);
            tmpB = zeros(8, 8);

            % Loop over all input states
            for k = 1:size(states,2)
                psi = states(:,k);
                rho = psi * psi';  

                % Fidelity operator for clone A:
                % conj(rho) (input) ⊗ rho (clone A) ⊗ I (clone B)
                tmpA = tmpA + kron(kron(conj(rho), rho), eye(2));

                % Fidelity operator for clone B:
                % conj(rho) (input) ⊗ I ⊗ rho
                tmpB = tmpB + kron(kron(conj(rho), eye(2)), rho);
            end

            % Normalize by number of input states
            obj.Omega_A = tmpA / size(states,2);
            obj.Omega_B = tmpB / size(states,2);
        end


        %% ------------------------------------------------------------
        %  Solve the primal SDP
        %
        %  maximize    Tr(E * Target)
        %  subject to  E >= 0
        %              Tr_{outputs}(E) = I
        %
        %  where Target = λ Ω_A + (1 − λ) Ω_B.
        %
        %  E is the 8×8 Choi matrix of the cloning channel.
        % ------------------------------------------------------------
        function [optimum, E] = solvePrimal(obj)

            % Weighted fidelity operator
            Target = obj.lambda*obj.Omega_A + (1 - obj.lambda)*obj.Omega_B;

            cvx_begin sdp quiet
                variable E(8, 8) hermitian

                maximize( real(trace(E * Target)) )

                subject to
                    % (1) Complete positivity
                    E >= 0;

                    % (2) Trace preservation:
                    %     Tr_{outputs}(E) = I_input  (outputs are subsystems 2 and 3)
                    PartialTrace(E, [2, 3], [2, 2, 2]) == eye(2);
            cvx_end

            optimum = cvx_optval;
        end



        %% ------------------------------------------------------------
        %  Solve the dual SDP
        %
        %  Dual program:
        %
        %    minimize   Tr(Y)
        %    subject to kron(Y, I_4) >= Target
        %
        %  where Y is a 2×2 Hermitian matrix.
        %
        %  Strong duality holds (Slater's condition satisfied), so the
        %  primal and dual optima coincide.
        % ------------------------------------------------------------
        function [optimum, Y] = solveDual(obj)

            Target = obj.lambda*obj.Omega_A + (1 - obj.lambda)*obj.Omega_B;

            cvx_begin sdp quiet
                variable Y(2, 2) hermitian

                minimize( real(trace(Y)) )

                subject to
                    % Only dual constraint:
                    % Lagrange matrix inequality for CP constraint
                    kron(Y, eye(4)) >= Target;
            cvx_end

            optimum = cvx_optval;
        end
    end
end