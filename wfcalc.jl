using Base.Threads
using Profile
using ThreadPools
using ProgressMeter
using LinearRegression
using DelimitedFiles

println("Using $(Threads.nthreads()) cores...")

if Threads.nthreads() == 1
    include("./wfreader.jl")
else
    include("./wfreader_thread.jl")
end
include("./coords.jl")
using .WaveFuncReaders, .CoordsTools

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

    #println(funcArray)


    wfnMat = zeros(Float64, (nMOs, nPrims))
    GTFvalMOs = zeros(Float64, nMOs)

    ThreadPools.@qthreads for j in 1:nPrims
        expx, expy, expz = type2exp[funcArray[j].functype]
        centerCoords = geom.points[funcArray[j].center].coords

        exponent = funcArray[j].exp

        relx, rely, relz = coords - centerCoords
        rr = relx^2 + rely^2 + relz^2
        expTerm = exp(-exponent * rr)

        GTFval = relx^expx * rely^expy * relz^expz * expTerm

        #println(GTFval)

        for i in 1:nMOs
            wfnMat[i,j] = primMatrix[i,j] * GTFval
        end
    end

    ThreadPools.@qthreads for i in 1:nMOs
        GTFvalMOs[i] = sum(wfnMat[i,:])
    end

    return GTFvalMOs
end

function genPoints(geom::Main.WaveFuncReaders.CoordsTools.Geometry, dLevel::Float64 = 1.5, ratiometric::Bool = false, resLevel::Int64 = 2)
    # dLevel stands for the scale of the resulted box; 
    # If ratiometric is true, the strides of different axis are ratiometric;
    # ResLevel stands for the number of sampling points per nanometer.

    println("\nGenerating space points from geometry...")

    coords = hcat(tmap(x -> geom.points[x].coords, 1:length(geom.atomNumbers))...)
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

    
    ThreadPools.@qthreads for i in 1:xNo
        for j in 1:yNo
            for k in 1:zNo
                spaceMat[i, j, k, 1] = xRange[i]
                spaceMat[i, j, k, 2] = yRange[j]
                spaceMat[i, j, k, 3] = zRange[k]
            end
        end
    end
    
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
    MOocc::Vector{Float64}, 
    MOenergy::Vector{Float64}, 
    primMatrix::Matrix{Float64},
    MO::Int64
    )

    spaceMat = genPoints(geom)
    println(size(spaceMat))
    xNo, yNo, zNo, dimensions = size(spaceMat)

    #noMOs = length(MOocc)

    wfn = zeros(Float64, (xNo, yNo, zNo))
    println("\nCalculating values of wavefunction in $(xNo * yNo * zNo) for the molecular orbital $MO (HOMO-$(length(MOocc) - MO))")

    progress = Progress(xNo)
    ThreadPools.@qthreads for i in 1:xNo
        for j in 1:yNo
            for k in 1:zNo
                wfn[i,j,k] = calcMOwfn(geom, funcArray, primMatrix, spaceMat[i,j,k,:])[MO]
            end
        end
        next!(progress)
    end

    finish!(progress)
    return wfn
end

function fullSpaceDens(
    geom::Main.WaveFuncReaders.CoordsTools.Geometry, 
    funcArray::Vector{Basis}, 
    MOocc::Vector{Float64}, 
    MOenergy::Vector{Float64}, 
    primMatrix::Matrix{Float64}
    )

    spaceMat = genPoints(geom)
    println(size(spaceMat))
    xNo, yNo, zNo, dimensions = size(spaceMat)

    #noMOs = length(MOocc)

    wfn = zeros(Float64, (xNo, yNo, zNo))
    println("Calculating values of electronic density in $(xNo * yNo * zNo) positions")

    progress = Progress(xNo)
    ThreadPools.@qthreads for i in 1:xNo
        for j in 1:yNo
            for k in 1:zNo
                wfn[i,j,k] = sum(calcMOwfn(geom, funcArray, primMatrix, spaceMat[i,j,k,:]))
            end
        end
        next!(progress)
    end

    finish!(progress)
    return wfn
end

function planeWFN(
    geom::Main.WaveFuncReaders.CoordsTools.Geometry, 
    funcArray::Vector{Basis}, 
    MOocc::Vector{Float64}, 
    MOenergy::Vector{Float64}, 
    primMatrix::Matrix{Float64}
    )

    plane = genPlanePoints(geom)
    nDotsX, nDotsY = size(plane)

    wfn = zeros(Float64, (nDotsX, nDotsY))
    progress = Progress(nDotsX)
    
    ThreadPools.@qthreads for I in CartesianIndices(plane)
        wfn[I] = sum(calcMOwfn(geom, funcArray, primMatrix, plane[I]))
    end

    return wfn
end

function main()
    geom, funcArray, MOocc, MOenergy, primMatrix, virial, totalEnergy = readWfn("hr.wfn")

    wfn = planeWFN(geom, funcArray, MOocc, MOenergy, primMatrix)

    writedlm("wfn.csv", wfn, "\t")
end

@time main()