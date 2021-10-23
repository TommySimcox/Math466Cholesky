# Thomas Simcox
# Project 1
# Math 466
# September, 2021

using LinearAlgebra;

function isPositiveDefinite(M::Matrix{Int64})::Bool
    return eigvals(M)[1] > 0 && eigvals(M)[2] > 0 && M == M' ? true : false;
end

function chol(M::Matrix{Int64})::Union{Matrix{Float64}, Nothing}
    # if the matrix isn't positive definite, return log indicating so
    !isPositiveDefinite(M) && return println("Matrix is not positive definite.")

    # destructure dims of array into variables
    (m, n) = size(M)

    # create empty matrix consisting of floats
    U = zeros(Float64, m, n)
    j = 1

    # calculate Cholesky decomposition
    U[j, j] = sqrt(M[j, j])
    for col in eachcol(M)
        for (i, v) in enumerate(col)
            # this is just a series of if checks that determines the calculation we need to do based on
            # whether i < j, i > j, or i === j
            i > j ? U[i, j] = v / U[j, j] : i < j ? U[i, j] = 0 : i === j && i === 2 ? U[i, j] = sqrt(v - (U[i, j-1])^2) : ""
        end
        j += 1
    end
    # return our new matrix U in upper-triangular form
    z = U[1, 2]
    U[1, 2] = U[2, 1]
    U[2, 1] = z
    return U
end


# test matrices
M_TEST = [[1, 2] [2, 13]];
M_1 = [[1, 0] [0, 4]];
M_2 = [[1, 0] [0, 0]];
M_3 = [[2 ,1] [1, 2]];
M_4 = [[2, 1] [3, 4]];
M_5 = [[1, 2] [2, 3]];
M_6 = [[5, -1] [-1, 3]];

# bigger test matrices from prog1c.dat
D_1 = [[14, 32, 53] [32, 77, 128] [53, 128, 213]]
D_2 = [[14, 32, 50] [32, 77, 122] [50, 122, 194]]
D_3 = [[1, 2, 3, 4] [2, 29, 36, 43] [3, 36, 109, 126] [4, 43, 126, 246]]
D_4 = [[1.50040, 1.06180, 1.37343, 0.75486, 0.78857] [1.06180, 1.45784, 1.41150, 0.87909, 1.20866] [1.37343, 1.41150, 2.16213, 0.79598, 1.17137] [0.75486, 0.87909, 0.79598, 0.73324, 0.74008] [0.78857, 1.20866, 1.17137, 0.74008, 1.10941]]
D_5 = [[1.63947, 1.10207, 0.60042, 0.87735, 0.24427] [1.10207, 0.94350, 1.36237, 1.00498, 0.47145] [0.60042, 1.36237, 1.42103, 0.66141, 1.34568] [0.87735, 1.00498, 0.66141, 1.11120, 0.99950] [0.24427, 0.47145, 1.34568, 0.99950, 0.32154]]
D_6 = [[0.00000, 0.50002, -0.02633, -0.23598, -0.16588] [-0.50002, 0.00000, -0.50740, 0.89786, 0.00493] [0.02633, 0.50740, 0.00000, 0.18580, 0.07650] [0.23598, -0.89786, -0.18580, 0.00000, -0.52940] [0.16588, -0.00493, -0.07650, 0.52940, 0.00000]]
D_7 = [[1.50040, 1.06180, 1.37343, 0.75486, 0.78857] [-1.06180, 1.45784, 1.41150, 0.87909, 1.20866] [-1.37343, -1.41150, 2.16213, 0.79598, 1.17137] [-0.75486, -0.87909, -0.79598, 0.73324, 0.74008] [-0.78857, -1.20866, -1.17137, -0.74008, 1.10941]]
D_8 = [[0.3161033, 0.3225228, 0.0457900, 0.1272576, -0.1258524, -0.0193280] [0.3225228, 0.5798926, -0.1325807, 0.0997750, -0.0012549, 0.2261822] [0.0457900, -0.1325807, 0.7555349, -0.0510603, -0.3411624, -0.4813482] [0.1272576, 0.0997750, -0.0510603, 0.5284544, -0.3153496, 0.2972834] [-0.1258524, -0.0012549, -0.3411624, -0.3153496, 0.7559030, -0.0408033] [-0.0193280, 0.2261822, -0.4813482, 0.2972834, -0.0408033, 0.6595527]]

# container for big matrices in prog1.dat
bigMatrices = [D_1, D_2, D_3, D_4, D_5, D_6, D_7, D_8]

# if matrix has cholesky decomp., display it
map(x -> try display(cholesky(x)) catch e println("Matrix not pos-def.") end, bigMatrices);

