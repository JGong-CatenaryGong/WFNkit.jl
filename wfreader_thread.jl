module WaveFuncReaders

    using Mendeleev
    using Mendeleev: elements
    using Unitful
    using InteractiveUtils
    using Base.Threads
    using ProgressMeter
    using LinearAlgebra

    export Geometry, Point, distance, angle, torsion, readWfn, Basis, Atom

    mutable struct Point
        coords::Vector{Float64}
    end

    mutable struct Geometry
        atomNumbers::Vector{Int8}
        atomMass::Vector{Float64}
        points::Vector{Point}
    end

    mutable struct Atom
        charge::Float64
        serial::Int64
        coords::Vector{Float64}
    end

    mutable struct Basis
        center::Int64
        functype::Int64
        exp::Float64
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

    function parseD(str::SubString{String}) # read scientific notation of float number with D instead of E
        if 'E' in str
            main, exp = split(str, 'E')
            new = main * 'e' * exp
            return parse(Float64, new)
        else
            main, exp = split(str, 'D')
            new = main * 'e' * exp
            return parse(Float64, new)
        end
    end

    function readWfn(filename::String)

        println("===WFNreader(Multi-thread) by JUN (01/18/2024)===")

        funcDict = Dict("GAUSSIAN" => 1, "SLATER" => 2)

        open(filename, "r") do f
            lines = readlines(f)

            title = lines[1] # title line

            basicInfo = split(lines[2]) # basic info line
            funcType = funcDict[basicInfo[1]]
            nOMO = parse(Int64, basicInfo[2])
            nGfunc = parse(Int64, basicInfo[5])
            nAtoms = parse(Int64, basicInfo[7])

            println("Reading basic information: $nOMO MOs, $nGfunc basis, $nAtoms atoms.")

            iLinesAtoms = lines[3:(2 + nAtoms)]

            atomArray = Array{Atom}(undef, nAtoms)
            for i in 1:nAtoms
                atomInfo = split(iLinesAtoms[i])
                aCharge = parse(Float64, atomInfo[end])
                aEle = atomInfo[1]
                aSerial = parse(Int64, atomInfo[2])

                coords = zeros(Float64, 3)
                j = 1
                for matches in eachmatch(r"-?[0-9|.]{4,}", iLinesAtoms[i]) # x,y,z coords
                    coords[j] = parse(Float64, matches.match)
                    j = j + 1
                end

                atomArray[i] = Atom(aCharge, aSerial, coords)
            end

            # Gen a geometry object
            points = map(c -> Point(c.coords), atomArray)
            atomNumbers = map(c -> convert(Int64, c.charge), atomArray)
            atomMass = map(i -> elements[i].atomic_mass / 1.0u"u", atomNumbers)
            geom = Geometry(atomNumbers, atomMass, points)

            centerLines = []
            typeLines = []
            expLines = []
            MOLines = []
            enerLine = []
            otherLines = []
            for line in lines[(2 + nAtoms):end]
                if occursin("CENTRE ASSIGNMENTS", line)
                    append!(centerLines, [line])
                elseif occursin("TYPE ASSIGNMENTS", line)
                    append!(typeLines, [line])
                elseif occursin("EXPONENTS", line)
                    append!(expLines, [line])
                elseif occursin("MO", line) || occursin(r"[-]?[0-9][.][0-9]+[E|D][+|-][0-9]{2}", line)
                    if occursin("MOSPIN", line)

                    else
                        append!(MOLines, [line])
                    end
                elseif occursin("VIRIAL", line) 
                    append!(enerLine, [line])
                else
                    append!(otherLines, [line])
                end
            end
            #println(centerLines)

            funcArray = Array{Basis}(undef, nGfunc)

            centerAssign = []
            for centerLine in centerLines
                #println(centerLine)
                for matches in eachmatch(r"S\s\s|[\s|0-9][\s|0-9][0-9]", centerLine)
                    if endswith(matches.match, " ")

                    else
                        append!(centerAssign, parse(Int64, matches.match))
                    end
                end
            end
            println("Loading $(size(centerAssign)) centers...")

            typeAssign = []
            for typeLine in typeLines
                #println(centerLine)
                for matches in eachmatch(r"[0-9]+", typeLine)
                    append!(typeAssign, parse(Int64, matches.match))
                end
            end
            println("Loading $(size(typeAssign)) function types...")

            exps = []
            for expLine in expLines
                for matches in eachmatch(r"[0-9][.][0-9]+[D|E][+|-][0-9]{2}", expLine)
                    append!(exps, parseD(matches.match))
                end
            end
            println("Loading $(size(exps)) exponents...")

            for k in 1:nGfunc
                funcArray[k] = Basis(centerAssign[k], typeAssign[k], exps[k])
            end
            #println(funcArray)

            println("Generating exponents matrix...")

            MOocc = zeros(Float64, nOMO)
            MOenergy = zeros(Float64, nOMO)
            primMatrix = zeros(Float64, (nOMO, nGfunc))

            MOprim = []
            progress = Progress(length(MOLines))
            lineNo = 1
            for line in MOLines
                if occursin("MO", line)
                    MOocc[lineNo] = parse(Float64, split(line)[end-4])
                    MOenergy[lineNo] = parse(Float64, split(line)[end])
                    lineNo = lineNo + 1
                else
                    for matches in eachmatch(r"[-]?[0-9][.][0-9]+[E|D][+|-][0-9]{2}", line)
                        append!(MOprim, parseD(matches.match))
                    end
                end
                next!(progress)
                
            end
            primMatrix = convert(Matrix{Float64}, transpose(reshape(MOprim, (nGfunc, nOMO))))
            finish!(progress)
            #println(primMatrix[2,1])

            #println(MOenergy)
            #println(enerLine)
            virEner = []
            for matches in eachmatch(r"[-]?[0-9][.][0-9]+", enerLine[1])
                append!(virEner, parse(Float64, matches.match))
            end

            virial = virEner[1]
            totalEnergy = virEner[2]

            return geom, funcArray, MOocc, MOenergy, primMatrix, virial, totalEnergy
        end
    end

end

