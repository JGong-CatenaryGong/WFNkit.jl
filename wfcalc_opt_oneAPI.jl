
if Threads.nthreads() == 1
    include("./wfreader.jl")
else
    include("./wfreader_thread.jl")
end
include("./coords.jl")
include("./numjy.jl")

using Base.Threads
using Profile, ProgressMeter
using LinearRegression
using DelimitedFiles
using .WaveFuncReaders, .CoordsTools, .Numjy
using oneAPI
using LinearAlgebra
using PyCall
using FLoops

function calcMOwfn(
    geom::Main.WaveFuncReaders.CoordsTools.Geometry, 
    funcArray::Vector{Basis}, 
    # MOocc::Vector{Float64}, 
    # MOenergy::Vector{Float64}, 
    primMatrix::Matrix{Float64}, 
    coords::Vector{Float64}
    )

    type2exp = Dict(
        1 => [0, 0, 0],
        2 => [1, 0, 0],
        3 => [0, 1, 0],
        4 => [0, 0, 1],
        5 => [2, 0, 0],
        6 => [0, 2, 0],
        7 => [0, 0, 2],
        8 => [1, 1, 0],
        9 => [1, 0, 1],
        10 => [0, 1, 1],
        11 => [3, 0, 0], 
        12 => [0, 3, 0],
        13 => [0, 0, 3],
        14 => [2, 1, 0],
        15 => [2, 0, 1],
        16 => [0, 2, 1],
        17 => [1, 2, 0],
        18 => [1, 0, 2],
        19 => [0, 1, 2], 
        20 => [1, 1, 1],
        21 => [0, 0, 4],
        22 => [0, 1, 3],
        23 => [0, 2, 2],
        24 => [0, 3, 1],
        25 => [0, 4, 0],
        26 => [1, 0, 3],
        27 => [1, 1, 2],
        28 => [1, 2, 1],
        29 => [1, 3, 0],
        30 => [2, 0, 2],
        31 => [2, 1, 1],
        32 => [2, 2, 0],
        33 => [3, 0, 1],
        34 => [3, 1, 0],
        35 => [4, 0, 0] 
    )

    atoms = geom.atomNumbers
    funcs = funcArray
    nMOs, nPrims = size(primMatrix)

    function getGTFval(funcElement::Main.WaveFuncReaders.Basis)
        
        expx, expy, expz = type2exp[funcElement.functype]
        centerCoords = geom.points[funcElement.center].coords
    
        exponent = funcElement.exp
    
        relx, rely, relz = coords - centerCoords
        rr = relx^2 + rely^2 + relz^2
        expTerm = exp(-exponent * rr)
    
        GTFval = relx^expx * rely^expy * relz^expz * expTerm
        return GTFval
    end
    #println(funcArray)

    if oneAPI.functional() #oneAPI check
        wfnMat = oneAPI.zeros(Float32, (nMOs, nPrims))
        GTFvalMOs = oneAPI.zeros(Float32, nMOs)

        GTFvals = oneAPI.zeros(Float32, nPrims)



        GTFvals = getGTFval.(funcArray)
        #println(size(GTFvals))

        GTFnewVals = reshape(GTFvals, (1, length(GTFvals)))

        wfnMat = GTFnewVals .* primMatrix

        GTFvalMOs = sum.(wfnMat)
    else
        wfnMat = zeros(Float64, (nMOs, nPrims))
        GTFvalMOs = zeros(Float64, nMOs)

        GTFvals = zeros(Float64, nPrims)



        GTFvals = getGTFval.(funcArray)
        #println(size(GTFvals))

        GTFnewVals = reshape(GTFvals, (1, length(GTFvals)))

        wfnMat = GTFnewVals .* primMatrix

        GTFvalMOs = sum.(wfnMat)
    end



    return GTFvalMOs
end

function calcGrad(
    geom::Main.WaveFuncReaders.CoordsTools.Geometry, 
    funcArray::Vector{Basis}, 
    # MOocc::Vector{Float64}, 
    # MOenergy::Vector{Float64}, 
    primMatrix::Matrix{Float64}, 
    coords::Vector{Float64},
    mode::Int64 = 2
    )

    #=
    mode: =2. calculate the gradient of electronic density; =1. calculate the gradient of wavefunction.
    =#

    type2exp = Dict(
        1 => [0, 0, 0],
        2 => [1, 0, 0],
        3 => [0, 1, 0],
        4 => [0, 0, 1],
        5 => [2, 0, 0],
        6 => [0, 2, 0],
        7 => [0, 0, 2],
        8 => [1, 1, 0],
        9 => [1, 0, 1],
        10 => [0, 1, 1],
        11 => [3, 0, 0], 
        12 => [0, 3, 0],
        13 => [0, 0, 3],
        14 => [2, 1, 0],
        15 => [2, 0, 1],
        16 => [0, 2, 1],
        17 => [1, 2, 0],
        18 => [1, 0, 2],
        19 => [0, 1, 2], 
        20 => [1, 1, 1],
        21 => [0, 0, 4],
        22 => [0, 1, 3],
        23 => [0, 2, 2],
        24 => [0, 3, 1],
        25 => [0, 4, 0],
        26 => [1, 0, 3],
        27 => [1, 1, 2],
        28 => [1, 2, 1],
        29 => [1, 3, 0],
        30 => [2, 0, 2],
        31 => [2, 1, 1],
        32 => [2, 2, 0],
        33 => [3, 0, 1],
        34 => [3, 1, 0],
        35 => [4, 0, 0] 
    )

    atoms = geom.atomNumbers
    funcs = funcArray
    nMOs, nPrims = size(primMatrix)

    function getGrad(funcElement::Main.WaveFuncReaders.Basis)
        
        i, j, k = type2exp[funcElement.functype]
        centerCoords = geom.points[funcElement.center].coords
    
        b = funcElement.exp
    
        x, y, z = coords - centerCoords
        rr = x^2 + y^2 + z^2
        
        #expTerm = exp(-exponent * rr)
        #GTFval = relx^expx * rely^expy * relz^expz * expTerm

        if mode == 1
            gradX = (i * x^(i-1) * y^j * z^k)/(exp(b * rr)) - (2 * b * x^(i+1) * y^j * z^k)/(exp(b * rr))
            gradY = (j * x^i * y^(j-1) * z^k)/(exp(b * rr)) - (2 * b * x^i * y^(j+1) * z^k)/(exp(b * rr))
            gradZ = (k * x^i * y^j * z^(k-1))/(exp(b * rr)) - (2 * b * x^i * y^j * z^(k+1))/(exp(b * rr))
        elseif mode == 2
            gradX = (2 * i * x^(2*i-1) * y^(2*j) * z^(2*k))/(exp(2*b * rr)) - (4 * b * x^(2*i+1) * y^(2*j) * z^(2*k))/(exp(2*b * rr))
            gradY = (2 * j * x^(2*i) * y^(2*j-1) * z^(2*k))/(exp(2*b * rr)) - (4 * b * x^(2*i) * y^(2*j+1) * z^(2*k))/(exp(2*b * rr))
            gradZ = (2 * k * x^(2*i) * y^(2*j) * z^(2*k-1))/(exp(2*b * rr)) - (4 * b * x^(2*i) * y^(2*j) * z^(2*k+1))/(exp(2*b * rr))
        end

        scaleVec = gradX, gradY, gradZ

        return scaleVec
    end

    if oneAPI.functional()
        #gradMat = oneAPI.zeros(Float64, (nMOs, nPrims))
        gradMOs = oneAPI.zeros(Float32, (nMOs))

        gradVals = oneAPI.zeros(Float32, nPrims)
        gradX = oneAPI.zeros(Float32, nPrims)
        gradY = oneAPI.zeros(Float32, nPrims)
        gradZ = oneAPI.zeros(Float32, nPrims)

        gradVals = getGrad.(funcArray)
        gradX = [x[1] for x in gradVals]
        gradY = [x[2] for x in gradVals]
        gradZ = [x[3] for x in gradVals]

        newGradX = reshape(gradX, (1, length(gradX)))
        newGradY = reshape(gradY, (1, length(gradY)))
        newGradZ = reshape(gradZ, (1, length(gradZ)))

        #newGradVals = reshape(gradScalars, (1, length(gradScalars)))

        gradMatX = newGradX .* primMatrix
        gradMatY = newGradY .* primMatrix
        gradMatZ = newGradZ .* primMatrix


        gradMOX = sum(gradMatX, dims = 2)
        gradMOY = sum(gradMatY, dims = 2)
        gradMOZ = sum(gradMatZ, dims = 2)
    else
        #gradMat = zeros(Float64, (nMOs, nPrims))
        gradMOs = zeros(Float64, (nMOs))

        gradVals = zeros(Float64, nPrims)
        gradX = zeros(Float64, nPrims)
        gradY = zeros(Float64, nPrims)
        gradZ = zeros(Float64, nPrims)

        gradVals = getGrad.(funcArray)
        gradX = [x[1] for x in gradVals]
        gradY = [x[2] for x in gradVals]
        gradZ = [x[3] for x in gradVals]

        newGradX = reshape(gradX, (1, length(gradX)))
        newGradY = reshape(gradY, (1, length(gradY)))
        newGradZ = reshape(gradZ, (1, length(gradZ)))

        #newGradVals = reshape(gradScalars, (1, length(gradScalars)))

        gradMatX = newGradX .* primMatrix
        gradMatY = newGradY .* primMatrix
        gradMatZ = newGradZ .* primMatrix

        gradMOX = sum(gradMatX, dims = 2)
        gradMOY = sum(gradMatY, dims = 2)
        gradMOZ = sum(gradMatZ, dims = 2)
    end

    return gradMOX, gradMOY, gradMOZ
end

function genPoints(geom::Main.WaveFuncReaders.CoordsTools.Geometry, dLevel::Float64 = 1.5, ratiometric::Bool = false, resLevel::Int64 = 5)
    # dLevel stands for the scale of the resulted box; 
    # If ratiometric is true, the strides of different axis are ratiometric;
    # ResLevel stands for the number of sampling points per nanometer.

    println("\nGenerating space points from geometry...")

    coords = hcat(map(x -> geom.points[x].coords, 1:length(geom.atomNumbers))...)
    xMax = maximum(coords[1,:])
    yMax = maximum(coords[2,:])
    zMax = maximum(coords[3,:])
    println("The top limit of axis: $xMax, $yMax, $zMax")

    xMin = minimum(coords[1,:])
    yMin = minimum(coords[2,:])
    zMin = minimum(coords[3,:])
    println("The bottom limit of axis: $xMin, $yMin, $zMin")

    scaleX = xMax - xMin   
    scaleY = yMax - yMin
    scaleZ = zMax - zMin

    centerX = xMin + scaleX / 2
    centerY = yMin + scaleY / 2
    centerZ = zMin + scaleZ / 2

    boxX = ceil(scaleX * dLevel)
    boxY = ceil(scaleY * dLevel)
    boxZ = ceil(scaleZ * dLevel)

    println("===============\nCenter coords: $centerX, $centerY, $centerZ\nThe system: $scaleX, $scaleY, $scaleZ\nThe box: $boxX, $boxY, $boxZ")

    stride = 1 / resLevel

    startX = centerX - boxX / 2
    startY = centerY - boxY / 2
    startZ = centerZ - boxZ / 2

    enX = centerX + boxX / 2
    enY = centerY + boxY / 2
    enZ = centerZ + boxZ / 2

    xRange = startX:stride:enX
    yRange = startY:stride:enY
    zRange = startZ:stride:enZ

    xNo = size(xRange)[1]
    yNo = size(yRange)[1]
    zNo = size(zRange)[1]

    spaceMat = zeros(Float64, (xNo, yNo, zNo, 3))

    Threads.@thread for I in CartesianIndices(spaceMat[:,:,:,1])
        spaceMat[I,1] = xRange[I[1]]
        spaceMat[I,2] = yRange[I[2]]
        spaceMat[I,3] = zRange[I[3]]
    end

    #=
    Threads.@threadfor i in 1:xNo
        for j in 1:yNo
            for k in 1:zNo
                spaceMat[i, j, k, 1] = xRange[i]
                spaceMat[i, j, k, 2] = yRange[j]
                spaceMat[i, j, k, 3] = zRange[k]
            end
        end
    end
    =#
    
    return spaceMat
end

function genPlanePoints(geom::Main.WaveFuncReaders.CoordsTools.Geometry, planeType::String = "opt", dLevel::Float64 = 1.5, resLevel::Int64 = 10)
    #=
    Generate a plane for wfn calculation.
    planeType:
    'opt' for the plane with a minimum average distance to all atoms;
    'x','y','z' for the plane parallel to the plane yz, plane xz and plane zy and cross the center atom
    =#

    println("\nGenerating $(planeType) plane points from geometry...")

    coords = hcat(tmap(x -> geom.points[x].coords, 1:length(geom.atomNumbers))...)
    xMax = maximum(coords[1,:])
    yMax = maximum(coords[2,:])
    zMax = maximum(coords[3,:])
    println("The top limit of axis: $xMax, $yMax, $zMax")

    xCoord = coords[1,:]
    yCoord = coords[2,:]
    zCoord = coords[3,:]
    xyArraay = hcat(xCoord, yCoord)

    xMin = minimum(coords[1,:])
    yMin = minimum(coords[2,:])
    zMin = minimum(coords[3,:])
    println("The bottom limit of axis: $xMin, $yMin, $zMin")

    scaleX = xMax - xMin   
    scaleY = yMax - yMin
    scaleZ = zMax - zMin

    centerX = xMin + scaleX / 2
    centerY = yMin + scaleY / 2
    centerZ = zMin + scaleZ / 2

    boxX = ceil(scaleX * dLevel)
    boxY = ceil(scaleY * dLevel)
    boxZ = ceil(scaleZ * dLevel)

    println("===============\nCenter coords: $centerX, $centerY, $centerZ\nThe system: $scaleX, $scaleY, $scaleZ\nThe box: $boxX, $boxY, $boxZ")

    stride = 1 / resLevel

    startX = centerX - boxX / 2
    startY = centerY - boxY / 2
    startZ = centerZ - boxZ / 2

    enX = centerX + boxX / 2
    enY = centerY + boxY / 2
    enZ = centerZ + boxZ / 2

    xRange = startX:stride:enX
    yRange = startY:stride:enY
    zRange = startZ:stride:enZ

    xNo = size(xRange)[1]
    yNo = size(yRange)[1]
    zNo = size(zRange)[1]
    
    if planeType == "opt"
        yRangeNew = range(startY, enY, xNo)

        xyRange = fill([0.0,0.0], (length(yRangeNew), length(yRangeNew)))
        for I in CartesianIndices(xyRange)
            xyRange[I] = [xRange[I[1]], yRangeNew[I[2]]]
        end
        println("Plane xys $(size(xyRange))")

        zRangeNew = zeros(Float64, (length(yRangeNew), length(yRangeNew)))
        println(size(zRangeNew))

        lr = linregress(xyArraay, zCoord)
        println("Fitted: $lr")
        for I in CartesianIndices(zRangeNew)
            zRangeNew[I] = lr(xyRange[I])
            #println(zRangeNew[I])
        end

        planeDots = append!.(xyRange, zRangeNew)    
        

    elseif planeType == "x"
        yRangeNew = range(startY, enY, xNo)
        zRangeNew = range(startZ, enZ, xNo)

        planeDots = fill([0.0,0.0,0.0], (xNo, xNo))

        for I in CartesianIndices(planeDots)
            #show(planeDots[I])
            planeDots[I] = [centerX, yRangeNew[I[1]], zRangeNew[I[2]]]
            
        end

        #println(size(planeDots))

    elseif planeType == "y"
        xRangeNew = range(startX, enX, yNo)
        zRangeNew = range(startZ, enZ, yNo)

        planeDots = fill([0.0,0.0,0.0], (yNo, yNo))

        for I in CartesianIndices(planeDots)
            #show(planeDots[I])
            planeDots[I] = [centerY, xRangeNew[I[1]], zRangeNew[I[2]]]
            
        end

        #println(size(planeDots))

    elseif planeType == "z"
        xRangeNew = range(startX, enX, zNo)
        yRangeNew = range(startY, enY, zNo)

        planeDots = fill([0.0,0.0,0.0], (zNo, zNo))

        for I in CartesianIndices(planeDots)
            #show(planeDots[I])
            planeDots[I] = [centerZ, xRangeNew[I[1]], yRangeNew[I[2]]]
            
        end

        #println(size(planeDots))
    end

    return planeDots
end

function fullSpaceWFN(
    geom::Main.WaveFuncReaders.CoordsTools.Geometry, 
    funcArray::Vector{Basis}, 
    primMatrix::Matrix{Float64}
    )

    spaceMat = genPoints(geom)
    println(size(spaceMat))
    xNo, yNo, zNo, dimensions = size(spaceMat)

    #noMOs = length(MOocc)

    wfn = zeros(Float64, (xNo, yNo, zNo))
    println("\nCalculating values of wavefunction in $(xNo * yNo * zNo) for the molecular orbital $MO (HOMO-$(length(MOocc) - MO))")

    progress = Progress(xNo * yNo * zNo)
    Threads.@thread for I in CartesianIndices(wfn)
        wfn[I] = sum(calcMOwfn(geom, funcArray, MOocc, primMatrix, spaceMat[I,:]))
        next!(progress)
    end

    finish!(progress)
    return wfn
end

function fullSpaceDens(
    geom::Main.WaveFuncReaders.CoordsTools.Geometry, 
    funcArray::Vector{Basis}, 
    # MOocc::Vector{Float64},
    primMatrix::Matrix{Float64},
    resLevel::Int64
    )

    spaceMat = genPoints(geom, 1.5, false, resLevel)
    println(size(spaceMat))
    xNo, yNo, zNo, dimensions = size(spaceMat)

    #noMOs = length(MOocc)

    if oneAPI.functional()
        wfn = zeros(Float32, (xNo, yNo, zNo))
    else
        wfn = zeros(Float64, (xNo, yNo, zNo))
    end
    println("Calculating values of electronic density in $(xNo * yNo * zNo) positions")

    progress = Progress(xNo * yNo * zNo)
    Threads.@thread for I in CartesianIndices(wfn)
        wfn[I] = sum(calcMOwfn(geom, funcArray, primMatrix, spaceMat[I,:]).^2)
        next!(progress)
    end

    #=
    progress = Progress(xNo)
    for i in 1:xNo
        for j in 1:yNo
            for k in 1:zNo
                wfn[i,j,k] = sum(calcMOwfn(geom, funcArray, MOocc, primMatrix, spaceMat[i,j,k,:]))
            end
        end
        next!(progress)
    end
    =#

    finish!(progress)
    return wfn
end

function planeWFN(
    geom::Main.WaveFuncReaders.CoordsTools.Geometry, 
    funcArray::Vector{Basis}, 
    primMatrix::Matrix{Float64}
    )

    plane = genPlanePoints(geom)
    nDotsX, nDotsY = size(plane)

    wfn = zeros(Float64, (nDotsX, nDotsY))
    progress = Progress(nDotsX)
    
    Threads.@thread for I in CartesianIndices(plane)
        wfn[I] = sum(calcMOwfn(geom, funcArray, primMatrix, plane[I]))
    end

    return wfn
end

function spaceGrad(
    geom::Main.WaveFuncReaders.CoordsTools.Geometry, 
    funcArray::Vector{Basis}, 
    primMatrix::Matrix{Float64}
)
    spaceMat = genPoints(geom)
    println(size(spaceMat))
    xNo, yNo, zNo, dimensions = size(spaceMat)

    #noMOs = length(MOocc)

    if oneAPI.functional()
        gradX = zeros(Float32, (xNo, yNo, zNo))
        gradY = zeros(Float32, (xNo, yNo, zNo))
        gradZ = zeros(Float32, (xNo, yNo, zNo))
        gradScalar = zeros(Float32, (xNo, yNo, zNo))
    else
        gradX = zeros(Float64, (xNo, yNo, zNo))
        gradY = zeros(Float64, (xNo, yNo, zNo))
        gradZ = zeros(Float64, (xNo, yNo, zNo))
        gradScalar = zeros(Float64, (xNo, yNo, zNo))
    end
    println("Calculating values of the gradients electronic density in $(xNo * yNo * zNo) positions")

    progress = Progress(xNo * yNo * zNo)
    Threads.@thread for I in CartesianIndices(gradX)
        gradX[I] = sum(calcGrad(geom, funcArray, primMatrix, spaceMat[I,:])[1])
        gradY[I] = sum(calcGrad(geom, funcArray, primMatrix, spaceMat[I,:])[2])
        gradZ[I] = sum(calcGrad(geom, funcArray, primMatrix, spaceMat[I,:])[3])
        gradScalar[I] = norm([gradX[I], gradY[I], gradZ[I]])
        next!(progress)
    end

    finish!(progress)
    return gradScalar
end

function spaceGradPy(
    geom::Main.WaveFuncReaders.CoordsTools.Geometry, 
    funcArray::Vector{Basis}, 
    # MOocc::Vector{Float64},
    primMatrix::Matrix{Float64}
    )

    spaceMat = genPoints(geom)
    println(size(spaceMat))
    xNo, yNo, zNo, dimensions = size(spaceMat)

    #noMOs = length(MOocc)

    if oneAPI.functional()
        wfn = zeros(Float32, (xNo, yNo, zNo))
    else
        wfn = zeros(Float64, (xNo, yNo, zNo))
    end

    println("Calculating values of electronic density in $(xNo * yNo * zNo) positions")

    progress = Progress(xNo * yNo * zNo)
    Threads.@thread for I in CartesianIndices(wfn)
        wfn[I] = sum(calcMOwfn(geom, funcArray, primMatrix, spaceMat[I,:]).^2)
        next!(progress)
    end

    #=
    progress = Progress(xNo)
    for i in 1:xNo
        for j in 1:yNo
            for k in 1:zNo
                wfn[i,j,k] = sum(calcMOwfn(geom, funcArray, MOocc, primMatrix, spaceMat[i,j,k,:]))
            end
        end
        next!(progress)
    end
    =#

    finish!(progress)
    #println(size(gradient(wfn)))
    densVecGradX, densVecGradY, densVecGradZ = Numjy.gradient(wfn)

    densGrad = [norm([densVecGradX[I], densVecGradY[I], densVecGradZ[I]]) for I in CartesianIndices(wfn)]

    println(size(cat(densVecGradX, densVecGradY, densVecGradZ; dims = 4)))

    #=
    hessX, hessY, hessZ = Numjy.gradient(densGrad)
    hess = [norm([hessX[I], hessY[I], hessZ[I]]) for I in CartesianIndices(wfn)]

    LinearAlgebra.eigvals!(hess)
    signMat = sign.(hess)

    wfnSigned = wfn .* signMat
    =#

    return densGrad, wfn
end

function getRDG(grad::Array{Float64}, dens::Array{Float64})
    val = 0.16162 * grad ./ (dens.^(4/3))
    return val
end

function getRDG(grad::Array{Float32}, dens::Array{Float32})
    val = 0.16162 * grad ./ (dens.^(4/3))
    return val
end

function main()
    println("Using $(Threads.nthreads()) cores...")
    println("oneAPI status: $(oneAPI.functional())")

    np = pyimport("numpy")


    geom, funcArray, MOocc, MOenergy, primMatrix, virial, totalEnergy = readWfn("tpe.wfn")

    for res in [1, 2, 4, 8, 10]

        dens = fullSpaceDens(geom, funcArray, primMatrix, res)
        np.save("tpe_dens_res$res.npy", dens, "\t")

    end

end

@time main()
