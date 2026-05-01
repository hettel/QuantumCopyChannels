classdef Utils

    methods (Access = private)
        function obj = Utils()
            % Privat Constructor
        end
    end

    methods (Static)
       function [kraus_ops, eigenvalues] = getKrausOperators(J, dim_in, dim_out, threshold)
            
            arguments
                J
                dim_in 
                dim_out
                threshold = 1e-9
            end
            
            dim_total = dim_in * dim_out;
            
            % Verify matrix dimensions
            if ~isequal(size(J), [dim_total, dim_total])
                error('Choi matrix E has incorrect dimensions. Expected %dx%d, got %dx%d', ...
                      dim_total, dim_total, size(J,1), size(J,2));
            end
            
            % Ensure E is Hermitian (should be from optimization)
            J = (J + J') / 2;
            
            % Eigenvalue decomposition of the Choi matrix
            [V, D] = eig( J );
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
   end 
end