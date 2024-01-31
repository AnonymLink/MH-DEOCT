module local_search
using DataFrames, CSV
using LinearAlgebra, Random
using TimerOutputs: @timeit, get_timer
using DecisionTree

# user defined modules:
if !("src/" in LOAD_PATH)
    push!(LOAD_PATH, "src/")
end   
using warmstart, oct

export local_searches

# [branch_size*a, branch_size*b], don't consider abd type, only consider OCT-a
function local_searches(funcion::Function, tree_da, X, Y, classes, tree_size, Nmin=1, cart=false)
    start = time()
    @timeit get_timer("Shared") "warmstart" begin
        var_number = 2
        # (X, y,tree_da, var_number, D = 4, prune_val=0.0)
        tree_depth = floor(Int64, log2(tree_size+1)-1)
        if cart == true
            println("Warm start with CART")
            initial_tree, ~ = warm_start_DT(X, Y, tree_da, var_number, Nmin, tree_depth)
        else
            initial_tree = rand(Float64, floor(Int64,tree_size/2)*var_number)
        end
        # println("Initial tree: ", initial_tree)
        best_fitness = funcion(tree_da, initial_tree, X, Y, classes, tree_size, Nmin, 0)
        best_best_fitness = best_fitness
        max_iter = 10 
        # In our tests, all the 65 datasets won't need more than 10 iterations. Most of them ended at the second iteration. 
        delta = Inf
        branch_size = floor(Int, tree_size/2)
        branch_set = [x for x in 1:branch_size]
        best_tree = copy(initial_tree)
        best_best_tree = copy(initial_tree)
    end
    @timeit get_timer("Shared") "searches" begin
        for iter in 1:max_iter
            branch_set = [x for x in 1:branch_size]
            while length(branch_set) > 0 # && delta > delta_limit
                branch_index = shuffle(branch_set)[1]
                branch_set = [x for x in branch_set if x != branch_index]
                X_i, Y_i = get_data(X, Y, branch_index, best_tree, branch_size, 2)
                # sort X by columns   
                sortX = zeros(Int64, size(X_i))    
                for i in 1:size(X_i,2)
                    sortX[:, i] = sortperm(X_i[:,i])
                end
                new_tree = optimize_node(funcion, tree_da, best_tree, branch_index, X_i, Y_i, classes, sortX)
                new_fitness = funcion(tree_da, new_tree, X, Y, classes, tree_size, Nmin, 0)
                # println("max_iter: ", max_iter, ", Branch index: ", branch_index, ", new fitness: ", new_fitness, ", best fitness: ", best_fitness)
                if new_fitness < best_fitness
                    delta = best_fitness - new_fitness
                    best_fitness = new_fitness
                    best_tree = copy(new_tree)
                end
            end
            println("Local search: ", iter, ", best fitness: ", best_fitness, ", best best fitness: ", best_best_fitness)
            if best_fitness < best_best_fitness
                println("Update - Local search: ", iter, ", best fitness: ", best_fitness, ", best best fitness: ", best_best_fitness)
                best_best_fitness = best_fitness
                best_best_tree = best_tree
            else
                println("Break - Local search: ", iter, ", best best fitness: ", best_best_fitness)
                break
            end
            if iter == max_iter
                println("Max_iter - Local search: ", iter, ", best best fitness: ", best_best_fitness)
            end
        end
    end
    used = time() - start
    # println("Time used: ", used)

    return best_tree, best_fitness
end

function optimize_node(funcion::Function, tree_da, initial_tree, branch_index, X, Y, classes, sortX, Nmin=1)
    var_number = 2
    sub_tree, sub_tree_l, sub_tree_u, childs = get_sub_tree(initial_tree, branch_index)
    # println("Sub tree: ", sub_tree, " ", sub_tree_l, " ", sub_tree_u, " ", childs)
    tree_size = floor(Int, (length(sub_tree)/var_number+1)*2-1)
    best_fitness = funcion(tree_da, sub_tree, X, Y, classes, tree_size, Nmin, 0)
    if length(sub_tree_l) != 0
        tree_size_l = floor(Int, (length(sub_tree_l)/var_number+1)*2-1)
        l_fitness = funcion(tree_da, sub_tree_l, X, Y, classes, tree_size_l, Nmin, 0)
        tree_size_u = floor(Int, (length(sub_tree_u)/var_number+1)*2-1)
        u_fitness = funcion(tree_da, sub_tree_u, X, Y, classes, tree_size_u, Nmin, 0)
    else
        l_fitness = Inf
        u_fitness = Inf
    end
    # split the node
    new_tree = best_split(funcion, tree_da, var_number, sub_tree, sub_tree_l, X, Y, classes, sortX)
    tree_size_n = floor(Int, (length(new_tree)/var_number+1)*2-1)
    n_fitness = funcion(tree_da, new_tree, X, Y, classes, tree_size_n, Nmin, 0)
    # compare new trees
    best_fitness, tree_index = findmin([best_fitness, l_fitness, u_fitness, n_fitness])
    # println("sub_tree: ", sub_tree, " ", sub_tree_l, " ", sub_tree_u, " ", new_tree)
    if tree_index == 1
        best_sub_tree = sub_tree
    elseif tree_index == 2
        best_sub_tree = sub_tree_l
    elseif tree_index == 3
        best_sub_tree = sub_tree_u
    else
        best_sub_tree = new_tree
    end

    best_tree = replace_sub_tree(initial_tree, best_sub_tree, childs)

    return best_tree
end

# E.g.
# branch_index = 2
# branch_depth = 1
# all_branch_depth = 3
# childs_l = [4, 8, 9]
# childs_u = [5, 10, 11]
# childs = [2, 4, 5, 8, 9, 10, 11]
function get_child_index(branch_index, all_branch_depth)
    branch_depth = floor(Int, floor(Int, log2(branch_index)))
    branch_size = 2^(all_branch_depth+1)-1
    childs_l = branch_index*2 <= branch_size ? Int64[branch_index*2] : Int64[]
    childs_u = branch_index*2+1 <= branch_size ? Int64[branch_index*2+1] : Int64[]
    for i in 1:all_branch_depth-branch_depth-1
        childs_l_i = Int64[x*2 for x in childs_l]
        childs_l_i = union(childs_l_i, Int64[x*2+1 for x in childs_l])
        childs_l = sort(union(childs_l, childs_l_i))
        childs_u_i = Int64[x*2 for x in childs_u]
        childs_u_i = union(childs_u_i, Int64[x*2+1 for x in childs_u])
        childs_u = sort(union(childs_u, childs_u_i))
    end
    childs = sort(union(Int64[branch_index], childs_l, childs_u))
    # println("childs: ", typeof(childs), " ", childs_l, " ", childs_u)
    return childs, childs_l, childs_u
end

function get_sub_tree(initial_tree, branch_index, var_number=2)
    branch_size = floor(Int, length(initial_tree)/var_number)
    all_branch_depth = floor(Int, log2(branch_size))
    childs, childs_l, childs_u = get_child_index(branch_index, all_branch_depth)
    # println("initial_tree: ", initial_tree)
    sub_tree = Float64[]
    sub_tree_l = Float64[]
    sub_tree_u = Float64[]
    for j in 1:var_number
        sub_tree = [sub_tree; Float64[initial_tree[(j-1)*branch_size + x] for x in childs]]
        sub_tree_l = [sub_tree_l; Float64[initial_tree[(j-1)*branch_size + x] for x in childs_l]]
        sub_tree_u = [sub_tree_u; Float64[initial_tree[(j-1)*branch_size + x] for x in childs_u]]
    end
    # println("Sub tree: ", sub_tree, " ", sub_tree_l, " ", sub_tree_u, " ", childs)
    return sub_tree, sub_tree_l, sub_tree_u, childs
end

function replace_sub_tree(initial_tree, best_sub_tree, childs)
    # println("Best sub tree: ", best_sub_tree)
    var_number = 2
    branch_size = floor(Int, length(initial_tree)/var_number)
    sub_branch_size = floor(Int, length(best_sub_tree)/var_number)
    best_tree = copy(initial_tree)
    best_tree[childs] .= 0
    best_tree[childs.+branch_size] .= 0

    for i in 1:sub_branch_size
        best_tree[childs[i]] = best_sub_tree[i]
        best_tree[childs[i]+branch_size] = best_sub_tree[i+sub_branch_size]
    end

    return best_tree
end

function best_split(funcion::Function, tree_da, var_number, sub_tree, sub_tree_l, X, Y, classes, sortX, Nmin=1)
    n, p = size(X)
    # classes = sort(unique(Y))
    K = length(classes)
    error = Inf
    best_sub_tree = sub_tree
    tree_size = floor(Int, (length(sub_tree)/var_number+1)*2-1)
    leaf_size = ceil(Int, tree_size/2)
    for j in 1:p
        z = Bool[]
        leaf_counts = zeros(Int, K, leaf_size)
        new_error = Inf
        Nmin_flag = 0
        for i in 1:(n-1)
            b = (X[sortX[i,j], j] + X[sortX[i+1,j], j])/2
            if i == 1
                new_sub_tree = replace_root_node(sub_tree, b, j, p) 
                new_error, Nmin_flag, z = funcion(tree_da, new_sub_tree, X, Y, classes, tree_size, Nmin, 3)
                if Nmin_flag > 0
                    Nmin_flag = true
                else
                    Nmin_flag = false
                end
                # z = [n,leaf_size] Nkt[i, t] = Nt[t] == 0 ? 0 : count(isequal(classes[i]), Y[z[:,t]])
                for l in 1:leaf_size
                    for k in 1:K
                        leaf_counts[k,l] = sum(z[:,l]) == 0 ? 0 : count(isequal(classes[k]), Y[z[:,l]])
                    end
                end
            else
                cur_sample = X[sortX[i,j], :]
                # assign from right to left
                ## right 
                l = findfirst(z[sortX[i,j], :])
                max_count, k = findmax(leaf_counts[:, l]) # old leaf
                leaf_counts[Y[sortX[i,j]], l] -= 1 # remove from old leaf
                new_max_count, new_k = findmax(leaf_counts[:, l]) # new leaf
                if new_k == k # no change in leaf
                    if Y[sortX[i,j]] != classes[k] # wrong classification in original
                        new_error = new_error - 1
                    end
                else # change in leaf
                    new_error = new_error - (sum(leaf_counts[:, l])-max_count)
                    new_error = new_error + (sum(leaf_counts[:, l])-new_max_count)
                end
                ## left
                l = oct_left(cur_sample, sub_tree_l, tree_da, var_number)
                max_count, k = findmax(leaf_counts[:, l]) # old leaf
                leaf_counts[Y[sortX[i,j]], l] += 1 # remove from old leaf
                new_max_count, new_k = findmax(leaf_counts[:, l]) # new leaf
                if new_k == k # no change in leaf
                    if Y[sortX[i,j]] != classes[k] # wrong classification in new
                        new_error = new_error + 1
                    end
                else # change in leaf
                    new_error = new_error - (sum(leaf_counts[:, l])-max_count)
                    new_error = new_error + (sum(leaf_counts[:, l])-new_max_count)
                end
                for l in 1:leaf_size
                    if !Nmin_flag && sum(leaf_counts[:, l]) < Nmin && sum(leaf_counts[:, l]) > 0
                        Nmin_flag = true
                    end
                end
            end
            #println("new_error: ", new_error, " ", error, " ", Nmin_flag)
            if !Nmin_flag && new_error < error
                error = new_error
                best_sub_tree = replace_root_node(best_sub_tree, b, j, p) 
            end
        end
    end

    return best_sub_tree
end

function oct_left(cur_sample, sub_tree_l, tree_da, var_number)
    p = length(cur_sample)
    branch_size = floor(Int, length(sub_tree_l)/var_number)
    a, b, d = trans_params_a(sub_tree_l, p, branch_size)
    l = 1
    while l <= branch_size
        if d[l] && cur_sample[a[:, l]][1] < b[l]
            l = l*2
        else
            l = l*2+1
        end
    end
    l = l-branch_size
    return l
end

function replace_root_node(sub_tree, b, j, p, var_number=2)
    branch_size = floor(Int, length(sub_tree)/var_number)
    new_sub_tree = copy(sub_tree)
    new_sub_tree[1] = (j-1) / (p)
    new_sub_tree[branch_size+1] = b
    return new_sub_tree
end

# tree_da - 1: tree_d; 2: tree_a; 3: tree_a_with_zero
function get_data(X, Y, i, xbest, size_branch, tree_da=1) #i=5
    # [floor(i/2), floor(i/4), ..., 1]
    ancesters = [floor(Int64, i/2^j) for j in floor(Int64, log2(i)):-1:0] # [1,2,5]
    size_ancs = length(ancesters) # 3
    n, p = size(X) 
    if tree_da == 1
        a, b, d = trans_params_d(xbest, p, size_branch) # [p, sizie_branch], [size_branch], [size_branch]  
    elseif  tree_da == 2
        a, b, d = trans_params_a(xbest, p, size_branch)
    elseif tree_da == 3
        a, b, d = trans_params_az(xbest, p, size_branch)
    end
    # println(a, b, d)
    selected = trues(n) 
    for i in 1:(size_ancs-1) # 1,2
        if ancesters[i+1] == 2*ancesters[i] # left_node, ax<b
            if d[ancesters[i]] == false
                selected .= false
            else
                for j in 1:n
                    if selected[j] && X[j, :][a[:, ancesters[i]]][1] >= b[ancesters[i]]
                        selected[j] = false
                    end
                end
            end
        else # right_node, ax>=b
            if d[ancesters[i]] == true
                for j in 1:n
                    if selected[j] && X[j, :][a[:, ancesters[i]]][1] < b[ancesters[i]]
                        selected[j] = false
                    end
                end
            end
        end            
    end

    return X[selected, :], Y[selected]
end

end # module