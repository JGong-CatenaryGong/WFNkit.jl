# The definition and standard of SMILES are from OpenSMILES community. http://opensmiles.org/

using Unicode
using PeriodicTable
using Graphs
using GraphPlot
using Compose
import Cairo, Fontconfig

mutable struct Molecule
    atoms::Vector{Int8}
    # bonds::Vector{Int8}
    connMat::Matrix{Float16}
end

function molToGraph(mol::Molecule)
    mat = mol.connMat
    vertices = mol.atoms
    no_vertices = length(vertices)

    g = SimpleGraph(mat)
end

function tokenize(smiles::String)
    # Convert serial SMILES to an array of tokens.

    len = length(smiles)
    # println(len)
    token = zeros(Int8, len)
    val = fill(' ', len)

    i = 1

    for char in smiles
        #println(char)
        charstr = string(char)
        if occursin(r"[A-Z]", charstr) # Non-aromatic atoms
            token[i] = 1
        elseif occursin(r"[a-z]", charstr) # Aromatic atoms or second letter of elements with two-letter names
            if occursin(r"[b,c,n,o,s,p]", charstr)
                token[i] = 2
            elseif occursin(r"se|as", charstr)
                token[i] = 2
            else
                token[i] = 3
            end
        elseif occursin(r"[0-9]", charstr) # Numbers of rings
            token[i] = 4
        elseif occursin(r"[=,#]", charstr) # Unsaturated bonds
            token[i] = 5
        elseif occursin(r"[\[,\]]", charstr) # Brackets
            token[i] = 6
        elseif occursin(r"[\(,\)]", charstr) # side chains
            token[i] = 7
        elseif occursin(r"[-,+]", charstr) # Charges
            token[i] = 8
        elseif occursin(r"[.]", charstr) # Non-bonded parts
            token[i] = 9
        else # other tokens
            token[i] = 10
        end
        val[i] = char
        i = i + 1
    end

    token, val
end

function readSmiles(smiles::String)
    token, val = tokenize(smiles)
    # println(token)

    no_atom = count(==(1), token) + count(==(2), token)
    atom_i = findall(x -> x <= 2, token)

    arom_i = findall(x -> x == 2, token)

    # index of atoms
    # println(atom_i)

    # println(atoms)
    symbols = [Symbol(uppercase(e)) for e = val[atom_i]]
    atoms = [elements[i].number for i = symbols]
    connMat = zeros(Float16, (no_atom, no_atom))

    no_bonds = convert(Int16, no_atom - 1 + count(==(4), token))
    # bonds = zeros(Int8, no_bonds)

    # possibly maximum number of bonds
    max_no_bonds = no_atom * (no_atom - 1)

    #= pre-allocated
    bond_a = zeros(Int8, max_no_bonds)
    bond_b = zeros(Int8, max_no_bonds)
    bond_order = ones(Float16, max_no_bonds)
    cursor = 1
    for j in atom_i
        if j == 1
            prev_val = ' '
            next_val = val[j + 1]
        elseif j == last(atom_i)
            prev_val = val[j - 1]
            next_val = ' '
        else
            # println(val[atom_i[cursor] + 1:atom_i[cursor + 1] - 1])
            interAtoms = token[atom_i[cursor] + 1:atom_i[cursor + 1] - 1]
            # println(interAtoms)

            prev_val = val[j - 1]
            next_val = val[j + 1]
 
            if all(i -> (i in [4, 6, 8]), interAtoms)
                if atom_i[cursor] ∈ arom_i && atom_i[cursor + 1] ∈ arom_i
                    bond_a[cursor] = j
                    bond_b[cursor] = atom_i[cursor + 1]
                    bond_order[cursor] = 1.5
                else
                    bond_a[cursor] = j
                    bond_b[cursor] = atom_i[cursor + 1]
                    bond_order[cursor] = 1
                end 
            end

        end
        if j + 1 ∈ atom_i #continual two atoms
            if j + 1 ∈ arom_i && j ∈ arom_i
                    bond_a[cursor] = j
                    bond_b[cursor] = atom_i[cursor + 1]
                    bond_order[cursor] = 1.5
            else
                bond_a[cursor] = j
                bond_b[cursor] = atom_i[cursor + 1]
                bond_order[cursor] = 1
            end                
        else
            if next_val == '('
                bond_a[cursor] = j
                bond_b[cursor] = atom_i[cursor + 1]
            elseif prev_val == '='
                bond_a[cursor] = j
                bond_b[cursor] = atom_i[cursor - 1]
                bond_order[cursor] = 2
            elseif prev_val == '#'
                bond_a[cursor] = j
                bond_b[cursor] = atom_i[cursor - 1]
                bond_order[cursor] = 3              
            end
        end

        cursor = cursor + 1
    end

    =#

    # Append

    bond_a = [0]
    bond_b = [0]
    bond_order = [0.0]
    cursor = 1
    for j in atom_i
        if j == 1
            prev_val = ' '
            next_val = val[j + 1]
        elseif j == last(atom_i)
            prev_val = val[j - 1]
            next_val = ' '
        else
            # println(val[atom_i[cursor] + 1:atom_i[cursor + 1] - 1])
            interAtoms = token[atom_i[cursor] + 1:atom_i[cursor + 1] - 1]
            # println(interAtoms)

            prev_val = val[j - 1]
            next_val = val[j + 1]
 
            if all(i -> (i in [4, 6, 8]), interAtoms)
                if atom_i[cursor] ∈ arom_i && atom_i[cursor + 1] ∈ arom_i
                    append!(bond_a, [j])
                    append!(bond_b, [atom_i[cursor + 1]])
                    append!(bond_order, [1.5])
                else
                    append!(bond_a, [j])
                    append!(bond_b, [atom_i[cursor + 1]])
                    append!(bond_order, [1])
                end 
            end

        end
        if j + 1 ∈ atom_i #continual two atoms
            if j + 1 ∈ arom_i && j ∈ arom_i
                append!(bond_a, [j])
                append!(bond_b, [atom_i[cursor + 1]])
                append!(bond_order, [1.5])
            else
                append!(bond_a, [j])
                append!(bond_b, [atom_i[cursor + 1]])
                append!(bond_order, [1])
            end                
        else
            if next_val == '('
                append!(bond_a, [j])
                append!(bond_b, [atom_i[cursor + 1]])
                append!(bond_order, [1])
            elseif prev_val == '='
                append!(bond_a, [j])
                append!(bond_b, [atom_i[cursor - 1]])
                append!(bond_order, [2])
            elseif prev_val == '#'
                append!(bond_a, [j])
                append!(bond_b, [atom_i[cursor - 1]])
                append!(bond_order, [3])             
            end
        end

        cursor = cursor + 1
    end

    num_i = findall(x -> x == 4, token)

    no_rings = convert(Int16, length(num_i) / 2)

    num_val = val[num_i]
    num_ring_no = parse.(Int16, val[num_i])

    for r in 1:no_rings
        #println(findfirst(x -> x == r, num_ring_no))
        first_i = num_i[findfirst(x -> x == r, num_ring_no)]
        #println(findlast(x -> x == r, num_ring_no))
        last_i = num_i[findlast(x -> x == r, num_ring_no)]
        prev_first_atom_i = findlast(x -> x < first_i, atom_i)
        prev_last_atom_i = findlast(x -> x < last_i, atom_i)

        append!(bond_a, atom_i[prev_first_atom_i[1]])
        append!(bond_b, atom_i[prev_last_atom_i[1]])
        append!(bond_order, [1])

    end
    
    println(bond_a)
    println(bond_b)
    

    for k in 1:length(bond_order)
        if bond_a[k] == 0

        elseif bond_b[k] == 0

        else
            a_i = findall(x -> x == bond_a[k], atom_i)[1]
            b_i = findall(x -> x == bond_b[k], atom_i)[1]
            connMat[a_i, b_i] = bond_order[k]
            connMat[b_i, a_i] = bond_order[k]
        end
    end

    println(connMat[2,21])

    Molecule(atoms, connMat)
end

test_smiles = "Cc12c(CCC(=O)[O-])ccc(CCCCCCCCC2)c1C(C#C)=NC(=O)CC.[Na+]"

@time mol = readSmiles(test_smiles)
print(mol)


@time g = molToGraph(mol)
println(mol)
println(g)

draw(PNG("./graph.png"), gplot(g))


