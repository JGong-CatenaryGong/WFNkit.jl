module WaveFuncReaders

export readWfn, Basis, Atom

using Mendeleev: elements
using Unitful
include("./coords.jl")
using .CoordsTools
using InteractiveUtils

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

function parseD(str::AbstractString) # read scientific notation of float number with D instead of E
    main, exp = split(str, 'D')
    new = main * 'e' * exp
    return parse(Float64, new)
end

function readWfn(filename::String)

    println("===WFNreader by JUN (01/18/2024)===")

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
        points = [Point(c.coords) for c = atomArray]
        atomNumbers = [convert(Int64, c.charge) for c = atomArray]
        atomMass = [elements[i].atomic_mass / 1.0u"u" for i in atomNumbers]
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
            elseif occursin("MO", line) || occursin(r"[0-9][.][0-9]+D[+|-][0-9]{2}", line)
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
            for matches in eachmatch(r"[0-9][.][0-9]+D[+|-][0-9]{2}", expLine)
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
        lineNo = 1
        for line in MOLines
            if occursin("MO", line)
                MOocc[lineNo] = parse(Float64, split(line)[end-4])
                MOenergy[lineNo] = parse(Float64, split(line)[end])
                lineNo = lineNo + 1
            else
                for matches in eachmatch(r"[0-9][.][0-9]+D[+|-][0-9]{2}", line)
                    append!(MOprim, parseD(matches.match))
                end
            end

            
        end
        primMatrix = convert(Matrix{Float64}, transpose(reshape(MOprim, (nGfunc, nOMO))))
        #println(primMatrix[2,1])

        #println(MOenergy)
        #println(enerLine)

        virial = parse(Float64, split(enerLine[1])[end])
        totalEnergy = parse(Float64, split(enerLine[1])[5])

        return geom, funcArray, MOocc, MOenergy, primMatrix, virial, totalEnergy
    end
end

end

