#using PeriodicTable


module CoordsTools

using Mendeleev
using Mendeleev: elements
using InteractiveUtils
using Unitful
using LinearAlgebra

export Geometry, Point, distance, angle, torsion

mutable struct Point
    coords::Vector{Float64}
end

mutable struct Geometry
    atomNumbers::Vector{Int8}
    atomMass::Vector{Float64}
    points::Vector{Point}
end

function readXYZ(filename::String)
    open(filename, "r") do f
        lines = readlines(f)
        allLines = [split(i) for i = lines]
        noAtoms = parse(Int8, allLines[1][1])
        atomLines = allLines[3:end]

        elementSeries = [Symbol(i[1]) for i in atomLines]
        atomNumbers = [elements[i].number for i in elementSeries]
        atomMass = [elements[i].atomic_mass / 1.0u"u" for i in elementSeries]

        pointsCoords = [Point([parse(Float64, i[2]), parse(Float64, i[3]), parse(Float64, i[4])]) for i in atomLines]

        geom = Geometry(atomNumbers, atomMass, pointsCoords)
        return geom
    end 
end

function distance(p1::Point, p2::Point)
    d = norm(p1.coords - p2.coords)
    return d
end

function angle(p1::Point, p2::Point, p3::Point)
    vec1 = p1.coords - p2.coords
    vec2 = p3.coords - p2.coords
    cosAngle = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    angle = 180 * acos(cosAngle) / π 

    return angle
end

function torsion(p1::Point, p2::Point, p3::Point, p4::Point)
    vec12 = p2.coords - p1.coords
    vec23 = p3.coords - p2.coords
    vec34 = p4.coords - p3.coords

    normVec123 = cross(vec12, vec23)
    normVec234 = cross(vec23, vec34)

    cosAngle = dot(normVec123, normVec234) / (norm(normVec123) * norm(normVec234))
    angle = 180 * acos(cosAngle) / π
    
    return angle
end

function genMatrix(geom::Geometry)
    coordinates = geom.points
    numberAtoms = length(geom.atomNumbers)
    distanceMatrix = zeros(Float64, (numberAtoms, numberAtoms))
    for i in 1:numberAtoms
        for j in 1:numberAtoms
            distanceMatrix[i, j] = distance(coordinates[i], coordinates[j])
        end
    end
    return distanceMatrix
end

function genVDWMat(geom::Geometry) # VDW radius
    coordinates = geom.points
    atomNo = geom.atomNumbers
    numberAtoms = length(geom.atomNumbers)
    distanceMatrix = zeros(Float64, (numberAtoms, numberAtoms))
    for i in 1:numberAtoms
        for j in 1:numberAtoms    
            if typeof(elements[atomNo[i]].vdw_radius) == Missing || typeof(elements[atomNo[j]].vdw_radius) == Missing
                distanceMatrix[i, j] = 0.0
            else
                distanceMatrix[i, j] = (elements[atomNo[i]].vdw_radius / 1u"pm" + elements[atomNo[j]].vdw_radius / 1u"pm") / 100
            end
        end
    end
    return distanceMatrix 
end

function genCovalMat(geom::Geometry, bondOrder::Int64) # Pyykko's covalent radius
    coordinates = geom.points
    atomNo = geom.atomNumbers
    numberAtoms = length(geom.atomNumbers)
    distanceMatrix = zeros(Float64, (numberAtoms, numberAtoms))
    for i in 1:numberAtoms
        for j in 1:numberAtoms
            if bondOrder == 1
                if typeof(elements[atomNo[i]].covalent_radius_pyykko) == Missing || typeof(elements[atomNo[j]].covalent_radius_pyykko) == Missing
                    distanceMatrix[i, j] = 0.0
                else
                    distanceMatrix[i, j] = (elements[atomNo[i]].covalent_radius_pyykko / 1u"pm" + elements[atomNo[j]].covalent_radius_pyykko / 1u"pm") / 100
                end
            elseif bondOrder == 2
                if typeof(elements[atomNo[i]].covalent_radius_pyykko_double) == Missing || typeof(elements[atomNo[j]].covalent_radius_pyykko_double) == Missing
                    distanceMatrix[i, j] = 0.0
                else
                    distanceMatrix[i, j] = (elements[atomNo[i]].covalent_radius_pyykko_double / 1u"pm" + elements[atomNo[j]].covalent_radius_pyykko_double / 1u"pm") / 100
                end
            elseif bondOrder == 3
                if typeof(elements[atomNo[i]].covalent_radius_pyykko_triple) == Missing || typeof(elements[atomNo[j]].covalent_radius_pyykko_triple) == Missing
                    distanceMatrix[i, j] = 0.0
                else
                    distanceMatrix[i, j] = (elements[atomNo[i]].covalent_radius_pyykko_triple / 1u"pm" + elements[atomNo[j]].covalent_radius_pyykko_triple / 1u"pm") / 100	
                end            
            end
        end
    end
    return distanceMatrix 
end

function determineBonds(geom::Geometry)
    vdwMat = genVDWMat(geom)
    singleMat = genCovalMat(geom, 1)
    doubleMat = genCovalMat(geom, 2)
    tripleMat = genCovalMat(geom, 3)

    distMat = genMatrix(geom)

    noAtoms = length(geom.atomNumbers)
    bondOrderMat = zeros(Float64, (noAtoms, noAtoms))
    for i in 1:noAtoms
        for j in 1:noAtoms
            if distMat[i,j] > singleMat[i,j] && distMat[i,j] <= vdwMat[i,j]
                bondOrderMat[i,j] = 0.5
            elseif doubleMat[i,j] < distMat[i,j] <= singleMat[i,j]
                bondOrderMat[i,j] = 1
            elseif tripleMat[i,j] < distMat[i,j] <= doubleMat[i,j]
                bondOrderMat[i,j] = 2
            elseif distMat[i,j] <= tripleMat[i,j]
                bondOrderMat[i,j] = 3
            else
                bondOrderMat[i,j] = 0
            end
        end
    end

    return bondOrderMat
end

function cartesianToInternal(geom::Geometry)
    coordinates = geom.points
    atomNo = length(geom.atomNumbers)

    distanceArray = zeros(Float64, atomNo)
    angleArray = zeros(Float64, atomNo)
    torsionArray = zeros(Float64, atomNo)

    for i in 1:atomNo
        if i > 1
            distanceArray[i] = distance(coordinates[i], coordinates[i-1])
        end
        if i > 2
            angleArray[i] = angle(coordinates[i], coordinates[i-1], coordinates[i-2])
        end
        if i > 3
            torsionArray[i] = torsion(coordinates[i], coordinates[i-1], coordinates[i-2], coordinates[i-3])
        end
    end

    return geom.atomNumbers, distanceArray, angleArray, torsionArray
end

#=
function main()
    filename = "pbo-md.xyz"
    g = readXYZ(filename)
    mat = cartesianToInternal(g)
    println(mat)
end

@time main()
=#

end

