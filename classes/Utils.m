classdef Utils

    methods (Access = private)
        function obj = Utils()
            % Privat Constructor
        end
    end

    methods (Static)
       function [kraus_ops, eigenvalues] = getKrausOperators(J, M, N, threshold)
            
            arguments
                J
                M 
                N
                threshold = 1e-9
            end

            dim_in    = 2^M;
            dim_out   = 2^N;
            dim_total = dim_in * dim_out;
            
            % Verify matrix dimensions
            if ~isequal(size(J), [dim_total, dim_total])
                error('Choi matrix E has incorrect dimensions. Expected %dx%d, got %dx%d', ...
                      dim_total, dim_total, size(J,1), size(J,2));
            end
            
            % Ensure E is Hermitian (should be from optimization)
            J = (J + J') / 2;
            
            % Eigenvalue decomposition of the Choi matrix
            [V, D] = eig( full(J) );
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




       
       function [is_valid, completeness_error] = verifyKrausOperators(kraus_ops, tol)
            
            arguments
                kraus_ops {mustBeA(kraus_ops, 'cell')}
                tol = 1e-8
            end
            
            if isempty(kraus_ops)
                is_valid = false;
                completeness_error = inf;
                return;
            end
            
            [dim_out, dim_in] = size(kraus_ops{1});
            
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




        function rho_out = applyChannel(kraus_ops, rho_in)
  
            arguments
                kraus_ops {mustBeA(kraus_ops, 'cell')}
                rho_in {mustBeNumeric}
            end
            
            [dim_out, dim_in] = size(kraus_ops{1});
            rho_out = zeros(dim_out, dim_out);
            
            for i = 1:length(kraus_ops)
                K = kraus_ops{i};
                rho_out = rho_out + K * rho_in * K';
            end
        end


        % Fibonacci function for creating evenly spaced random points on the Bloch sphere
        % Note: The created points are determinitic dependent on the parameter n
        function states = fibonacci_qubits(n)
            golden_angle = pi * (3 - sqrt(5));
            states = zeros(2,1, n);
            
            for i = 0:(n-1)
                % coordinates on the Bloch sphere
                z = 1 - 2*i/(n-1);
                r = sqrt(1 - z^2);
                phi = i * golden_angle;
                
                x = r * cos(phi);
                y = r * sin(phi);
                             
                % Qubit state
                theta = acos(z);
                varphi = atan2(y, x);
        
                states(:,:,i+1) = [cos(theta/2); exp(1i * varphi) * sin(theta/2)];
            end
        end

        % Create random unitary operation
        function rand_U = create_random_U()
             % random parameter for unitary operation
             lambda = rand*2*pi;
             theta  = rand*pi;
             phi    = rand*2*pi;
            
             U = [ cos(theta/2)       -exp(1i*lambda)*sin(theta/2); 
                   exp(1i*phi)*sin(theta/2)  exp(1i*(phi+lambda))*cos(theta/2) ];
        
             rand_U = U;
        end

        % Crerates a random qubit 
        function rand_qubit = create_random_qubit() 
             % Random parameters for Bloch sphere
             theta = acos(2*rand - 1);  % Uniform on sphere
             phi = 2 * pi * rand;
             state_init = [cos(theta/2); exp(1i*phi)*sin(theta/2)];
                     
             U = Utils.create_random_U();
        
             % create random state 
             rand_qubit = U*state_init;
        end

        % Crerates a random phase covraiant qubit 
        function rand_qubit = create_random_phase_covariant_qubit()  
             phi = 2 * pi * rand;
             rand_qubit = [1; exp(1i*phi)]/sqrt(2);
        end


        function states = create_entangled_fibonacci_states(n_states)

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
                  states_fibonacci{k} = Utils.generate_mbe_state( ...
                            theta1, phi1, psi1, theta2, phi2, psi2, gamma );
             end
        
             states = [ states_fibonacci{:} ];
        end
        
        % Helper function: Generate maximally entangled bipartite state
        function psi = generate_mbe_state(theta1, phi1, psi1, theta2, phi2, psi2, gamma)
        
            U_A = Utils.euler_to_unitary(theta1, phi1, psi1);
            U_B = Utils.euler_to_unitary(theta2, phi2, psi2);
        
            % Reference Bell state |Φ+⟩ = (|00⟩ + |11⟩) / sqrt(2)
            % with an additional adjustable phase e^{i γ} on |11⟩.
            bell = [1; 0; 0; exp(1i * gamma)] / sqrt(2);
        
            % Apply local unitaries: |ψ⟩ = (U_A ⊗ U_B) |Φ+⟩
            U = kron(U_A, U_B);
            psi = U * bell;
        
            % Normalize result (normally unnecessary but ensures numerical stability)
            psi = psi / norm(psi);
        end
        
        
        %Convert Euler angles (ZYZ) to SU(2) unitary
        function U = euler_to_unitary(theta, phi, psi)
        
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

        function states = create_maximal_entangled_random_states(n_states)
             % Preallocate cell array of pure states
             states_random = cell(n_states, 1);

             for k = 1:n_states
                 gamma = 2*pi*rand;
                 bell = [1; 0; 0; exp(1i * gamma)] / sqrt(2);

                 U_A = Utils.create_random_U;
                 U_B = Utils.create_random_U();

                 U = kron(U_A, U_B);
                 psi = U * bell;

                 states_random{k} = psi / norm(psi);
             end

             states = [ states_random{:} ];
        end

   end 
end