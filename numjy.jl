module Numjy

using LinearAlgebra

export gradient

function gradient(ar::Array{Float32})
    nDim = ndims(ar)
    #println(nDim)
    if nDim > 3
        println("Gradient model could not deal with more than 3 dimensions")
    elseif nDim == 1
        xMax = length(ar)
        gradMatX = copy(ar)
        for I in CartesianIndices(ar)
            if i == 1
                gradMatX[I] = (ar[i + 1, j] - ar[i, j]) / 2
            elseif i == xMax
                gradMatX[I] = (ar[i, j] - ar[i - 1, j]) / 2
            else
                gradMatX[I] = (ar[i + 1, j] - ar[i - 1, j]) / 2
            end
        end

        return gradMatX
    elseif nDim == 2
        xMax, yMax = size(ar)
        gradMatX = copy(ar)
        gradMatY = copy(ar)
        for I in CartesianIndices(ar)
            i = I[1]
            j = I[2]

            if i == 1
                gradMatX[I] = (ar[i + 1, j] - ar[i, j]) / 2
            elseif i == xMax
                gradMatX[I] = (ar[i, j] - ar[i - 1, j]) / 2
            else
                gradMatX[I] = (ar[i + 1, j] - ar[i - 1, j]) / 2
            end

            if j == 1
                gradMatY[I] = (ar[i, j + 1] - ar[i, j]) / 2
            elseif j == yMax
                gradMatY[I] = (ar[i, j] - ar[i, j - 1]) / 2
            else
                gradMatY[I] = (ar[i, j + 1] - ar[i, j - 1]) / 2
            end
        end

        return gradMatX, gradMatY
    elseif nDim == 3

        xMax, yMax, zMax = size(ar)
        gradMatX = copy(ar)
        gradMatY = copy(ar)
        gradMatZ = copy(ar)

        for I in CartesianIndices(ar)
            i = I[1]
            j = I[2]
            k = I[3]

            if i == 1
                gradMatX[I] = (ar[i + 1, j, k] - ar[i, j, k]) / 2
            elseif i == xMax
                gradMatX[I] = (ar[i, j, k] - ar[i - 1, j, k]) / 2
            else
                gradMatX[I] = (ar[i + 1, j, k] - ar[i - 1, j, k]) / 2
            end

            if j == 1
                gradMatY[I] = (ar[i, j + 1, k] - ar[i, j, k]) / 2
            elseif j == yMax
                gradMatY[I] = (ar[i, j, k] - ar[i, j - 1, k]) / 2
            else
                gradMatY[I] = (ar[i, j + 1, k] - ar[i, j - 1, k]) / 2
            end

            if k == 1
                gradMatZ[I] = (ar[i, j, k + 1] - ar[i, j, k]) / 2
            elseif k == zMax
                gradMatZ[I] = (ar[i, j, k] - ar[i, j, k - 1]) / 2
            else
                gradMatZ[I] = (ar[i, j, k + 1] - ar[i, j, k - 1]) / 2
            end

        end
        #println(size(gradMatY))

        return gradMatX, gradMatY, gradMatZ
    end
end

function gradient(ar::Array{Float64})
    nDim = ndims(ar)
    #println(nDim)
    if nDim > 3
        println("Gradient model could not deal with more than 3 dimensions")
    elseif nDim == 1
        xMax = length(ar)
        gradMatX = copy(ar)
        for I in CartesianIndices(ar)
            if i == 1
                gradMatX[I] = (ar[i + 1, j] - ar[i, j]) / 2
            elseif i == xMax
                gradMatX[I] = (ar[i, j] - ar[i - 1, j]) / 2
            else
                gradMatX[I] = (ar[i + 1, j] - ar[i - 1, j]) / 2
            end
        end

        return gradMatX
    elseif nDim == 2
        xMax, yMax = size(ar)
        gradMatX = copy(ar)
        gradMatY = copy(ar)
        for I in CartesianIndices(ar)
            i = I[1]
            j = I[2]

            if i == 1
                gradMatX[I] = (ar[i + 1, j] - ar[i, j]) / 2
            elseif i == xMax
                gradMatX[I] = (ar[i, j] - ar[i - 1, j]) / 2
            else
                gradMatX[I] = (ar[i + 1, j] - ar[i - 1, j]) / 2
            end

            if j == 1
                gradMatY[I] = (ar[i, j + 1] - ar[i, j]) / 2
            elseif j == yMax
                gradMatY[I] = (ar[i, j] - ar[i, j - 1]) / 2
            else
                gradMatY[I] = (ar[i, j + 1] - ar[i, j - 1]) / 2
            end
        end

        return gradMatX, gradMatY
    elseif nDim == 3

        xMax, yMax, zMax = size(ar)
        gradMatX = copy(ar)
        gradMatY = copy(ar)
        gradMatZ = copy(ar)

        for I in CartesianIndices(ar)
            i = I[1]
            j = I[2]
            k = I[3]

            if i == 1
                gradMatX[I] = (ar[i + 1, j, k] - ar[i, j, k]) / 2
            elseif i == xMax
                gradMatX[I] = (ar[i, j, k] - ar[i - 1, j, k]) / 2
            else
                gradMatX[I] = (ar[i + 1, j, k] - ar[i - 1, j, k]) / 2
            end

            if j == 1
                gradMatY[I] = (ar[i, j + 1, k] - ar[i, j, k]) / 2
            elseif j == yMax
                gradMatY[I] = (ar[i, j, k] - ar[i, j - 1, k]) / 2
            else
                gradMatY[I] = (ar[i, j + 1, k] - ar[i, j - 1, k]) / 2
            end

            if k == 1
                gradMatZ[I] = (ar[i, j, k + 1] - ar[i, j, k]) / 2
            elseif k == zMax
                gradMatZ[I] = (ar[i, j, k] - ar[i, j, k - 1]) / 2
            else
                gradMatZ[I] = (ar[i, j, k + 1] - ar[i, j, k - 1]) / 2
            end

        end
        #println(size(gradMatY))

        return gradMatX, gradMatY, gradMatZ
    end
end



end