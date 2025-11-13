include("constants.jl")
using ArgParse


"""
Best possible constructions: https://oeis.org/A006855/list

N:    1  2  3  4  5  6  7  8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34   35   36   37   38   39   40    
f(N): 0, 1, 3, 4, 6, 7, 9, 11, 13, 16, 18, 21, 24, 27, 30, 33, 36, 39, 42, 46, 50, 52, 56, 59, 63, 67, 71, 76, 80, 85, 90, 92, 96, 102, 106, 110, 113, 117, 122, 127

"""

#function parse_args()
#    s = ArgParseSettings()
#    @add_arg_table s begin
#        "-N", "--number"
#        help = "specifies the value of N"
#        arg_type = Int
#        default = 20  # default value if -N is not provided
#    end
#    return parse_args(s)
#end

#args = parse_args()
const N = 40 #get(args, :number, 20)#hy



function find_all_four_cycles(adjmat::Matrix{Int})
    N = size(adjmat, 1)
    four_cycles = Vector{Tuple{Int8, Int8, Int8, Int8}}()

    # Loop over all quadruples (a, b, c, d) where a < b < c < d
    for a in 1:N
        for b in a+1:N
            for c in a+1:N
                for d in b+1:N
                    if adjmat[a, b] == 1 && adjmat[b, c] == 1 && adjmat[c, d] == 1 && adjmat[d, a] == 1
                        push!(four_cycles, (a, b, c, d))
                    end
                end
            end
        end
    end

    return four_cycles
end




function convert_adjmat_to_string(adjmat::Matrix{Int})::String
    entries = []

    # Collect entries from the upper diagonal of the matrix
    for i in 1:N-1
        for j in i+1:N
            push!(entries, string(adjmat[i, j]))
        end
        #push!(entries,"2")#hy
        push!(entries,",")#hy
    end

    # Join all entries into a single string
    return join(entries)
end

function ordered(edge)::Tuple{Int, Int}
    if edge[1] <= edge[2]
        return edge
    end
    return (edge[2], edge[1])
end


#using PyCall
#pushfirst!(PyVector(pyimport("sys")."path"),"/Users/hyeyun/Experiment/PFD/transformers_math_experiments")
#js_data = pyimport("data")
#function greedy_search_from_startpoint(db, obj::OBJ_TYPE, additional_loops=0)::Vector{OBJ_TYPE}
#    """
#    Main greedy search algorithm.
#    It starts and ends with some construction
#    """
#
#    #CGSHOP26
#    input_path = "/Users/hyeyun/Experiment/PFD/CGSHOP26/data/benchmark_instances/"
#    input_file = input_path * "random_instance_4_40_2.json" #test
#
#    dt = js_data.Data(input_file)
#    # Compute Center
#    println("Center computed")
#    center, dist = dt.find_center()
#    N = center.num_pts
#    adjmat = zeros(Int, N, N)
#
#    # Perturbate Center
#    println("Center Perturbated")
#    center.random_flip(10)
#
#    # Compute sum(pfd(T,C))
#    dist, _ = dt.compute_center_dist(center)
#    center, dist = dt.random_move()
#
#    edges = center.edges
#    for edge in edges
#        i,j = edge[1]
#        i+=1
#        j+=1
#        adjmat[i,j] = 1
#        adjmat[j,i] = adjmat[i,j]
#    end
#    return [convert_adjmat_to_string(adjmat)]
#end

function generate_center(db, obj::OBJ_TYPE, additional_loops=0)::Vector{OBJ_TYPE}
    input_path = "/Users/hyeyun/Experiment/PFD/CGSHOP26/data/benchmark_instances/"
    Tris_file = input_path * "random_instance_4_40_2.json"


end

function reward_calc(obj::OBJ_TYPE)::REWARD_TYPE
    """
    Function to calculate the reward of a final construction
    (E.g. number of edges in a graph, etc)
    """
    return count(isequal('1'), obj)
end


function empty_starting_point()::OBJ_TYPE
    """
    If there is no input file, the search starts always with this object
    (E.g. empty graph, all zeros matrix, etc)
    """

    #adjmat = zeros(Int, N, N)
    #return convert_adjmat_to_string(adjmat)
    #hy
    #println("...empty_starting_point ")
    return "1"
end
