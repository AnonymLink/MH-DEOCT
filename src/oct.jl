module oct
using DataFrames, CSV
using LinearAlgebra, Random, StatsBase
using TimerOutputs: @timeit, get_timer

export OCT, trans_params_d, trans_params_a,trans_params_az, OCT_test

function trans_params_d(candidate, p, size_branch)
    #transform candidate solution ∈ ℜ^|branch_nodes|*3 to a = ajt ∈ {0,1}, b = bt ∈ ℜ^|branch_nodes|, and d = dt ∈ {0,1}:
    a = falses(p,size_branch) # this vectors help us to select which feature use at node t
    for t in 1:size_branch
        for feature in 1:p
            if feature != p
                if (candidate[t]>=(feature-1)/p) && (candidate[t]<(feature)/p)
                    a[feature,t] = true
                    break
                end
            else
                if (candidate[t]>=(feature-1)/p) && (candidate[t]<=1)
                    a[feature,t] = true
                    break
                end
            end
        end
    end            
    d = falses(1, size_branch) #set of nodes that apply a split
    for t in 1:size_branch
        # (5) enforce the hierarchical structure of the tree:
        if t>1        
            if candidate[t+size_branch]>candidate[floor.(Int, t/2)+size_branch]
                candidate[t+size_branch]=candidate[floor.(Int, t/2)+size_branch]
            end
        elseif t==1 && candidate[t+size_branch]<0.5 
            candidate[t+size_branch]=candidate[t+size_branch]+0.5
        end
        if candidate[t+size_branch]>=0.5 
            d[t] = true
        end
    end
    b = candidate[size_branch*2+1:size_branch*3]

    return a, b, d
end

# tree_a without zeros
function trans_params_a(candidate, p, size_branch)
    a = falses(p,size_branch) # this vectors help us to select which feature use at node t
    for t in 1:size_branch
        for feature in 1:p
            if feature != p
                if (candidate[t]>=(feature-1)/p) && (candidate[t]<(feature)/p)
                    a[feature,t] = true
                    break
                end
            else
                if (candidate[t]>=(feature-1)/p) && (candidate[t]<=1)
                    a[feature,t] = true
                    break
                end
            end
        end
    end            
    b = candidate[size_branch+1:size_branch*2]
    d = sum(a, dims=1) .> 0
    for t in 1:size_branch
        # (5) enforce the hierarchical structure of the tree:
        if t>1        
            if d[t]>d[floor.(Int, t/2)]
                d[t]=d[floor.(Int, t/2)]
            end
        else # root node
            d[t]=true
        end
    end
    return a, b, d
end

# tree_a_with_zeros
function trans_params_az(candidate, p, size_branch)
    a = falses(p,size_branch) # this vectors help us to select which feature use at node t
    # print("candidate:", candidate)
    for t in 1:size_branch
        for feature in 1:p
            if feature != p
                if (candidate[t]>=(feature-1)/(p+1)) && (candidate[t]<(feature)/(p+1))
                    a[feature,t] = true
                    break
                end
            else
                if (candidate[t]>=(feature-1)/(p+1)) && (candidate[t]<=p/(p+1))
                    a[feature,t] = true
                    break
                end
            end
        end
    end            
    b = candidate[size_branch+1:size_branch*2]
    d = sum(a, dims=1) .> 0
    for t in 1:size_branch
        # (5) enforce the hierarchical structure of the tree:
        if t>1        
            if d[t]>d[floor.(Int, t/2)]
                d[t]=d[floor.(Int, t/2)]
            end
        else # root node
            d[t]=true
        end
    end
    return a, b, d
end

# tree_da - 1: tree_d; 2: tree_a; 3: tree_a_with_zero
function OCT(tree_da, candidate, X, Y, classes, tree_size, Nmim, whatReturn, alpha=0.0)
    @timeit get_timer("Shared") "tree structure" begin
        # TREE STRUCTURE: ################################################################
        branch_nodes = [trunc(Int64, x) for x in 1:floor(tree_size/2)]     #branch nodes
        leaf_nodes = [trunc(Int64, x) for x in floor(tree_size/2)+1:tree_size]   #leaf nodes
        size_branch = size(branch_nodes,1)
        size_leaf = size(leaf_nodes,1)
        n = size(X,1)
        p = size(X,2)

        # classes=unique(Y)      #get a dataframe with the classes in the data
        K=size(classes,1)    #number of classes
        if tree_da == 1
            a, b, d = trans_params_d(candidate, p, size_branch)
        elseif tree_da == 2
            a, b, d = trans_params_a(candidate, p, size_branch)
        elseif tree_da == 3
            a, b, d = trans_params_az(candidate, p, size_branch)
            # println("a: ", a, " b: ", b, " d: ", d)
        end
    end # end of @timeit get_timer("Shared") "tree structure"

    @timeit get_timer("Shared") "class cost" begin
        # points assigned to each leaf node; zit = 𝟙 {𝐱i is in node t};
        # in this part, we track which 𝐱i arrives to t, ∀t ∈ leaf_nodes:   
        @timeit get_timer("Shared") "count" begin
            z = falses(n,tree_size)                      
            for i in 1:n                                   
                t=1                     #always starts in the root node
                # print(" t: ", t)
                while true
                    # println(X[i, :][a[:, t]])
                    if d[t] && X[i, :][a[:, t]][1] < b[t]
                        t = t*2
                        if t>size_branch
                            break
                        end
                        # print(" t: ", t)
                    else
                        t = t*2 + 1
                        if t>size_branch
                            break
                        end
                        # print(" t: ", t)
                    end
                end
                # println(" t: ", t)
                z[i,t]=true
            end
        end # end of @timeit get_timer("Shared") "count"

        @timeit get_timer("Shared") "cost" begin
            ct = zeros(Int64,1,tree_size) # Labels at node t
            Nt = zeros(Int64,1,tree_size) # Number of points at node t
            Nct = zeros(Int64,1,tree_size) # Number of points at node t with label ct[t]
            Lt = zeros(Int64,1,tree_size) # Cost of node t
            Nmin_flag = false # Nmin=0 for now
            for t in size_branch+1:tree_size
                # print(" t: ", t)
                Nt[t] = sum(z[:,t]) # Number of points at leaf node t
                # print(" sum(z[:,t]): ", Nt[t])
                if Nt[t] == 0
                    continue
                elseif Nt[t] < Nmim
                    Nmin_flag = true
                    continue
                end

                Yt = Y[z[:,t]]
                ct[t] = StatsBase.mode(Yt) # Label at leaf node t
                Nct[t] = count(isequal(ct[t]), Yt) # Number of points at leaf node t with label ct[t]
                Lt[t] = Nt[t] - Nct[t] # Cost of leaf node t
            end

            sumLt = sum(Lt) # sum of the cost of the leaf nodes
            octCost = sumLt
            
            if whatReturn==1 # used for debug case
                return d, a, ct
            elseif whatReturn==2 # used for test case
                return octCost
            elseif whatReturn==4 # used for training set in the test case
                return octCost, [x[1] for x in ct[size_branch+1:tree_size]], Nmin_flag
            end

            # complexity of the tree:
            sumdt = sum(d) * alpha

            # optimal classification tree cost:
            octCost = octCost + sumdt
            if whatReturn==3 # used for validating minleafsize for local_search
                return octCost, Nmin_flag, z[:, size_branch+1:tree_size] 
            end

        end # end of @timeit get_timer("Shared") "cost"
    end # end of @timeit get_timer("Shared") "class count"

    return octCost
end

function OCT_test(c, tree_da, candidate, X, Y, classes, tree_size)
    # TREE STRUCTURE: ################################################################
    branch_nodes = [trunc(Int64, x) for x in 1:floor(tree_size/2)]     #branch nodes
    leaf_nodes = [trunc(Int64, x) for x in floor(tree_size/2)+1:tree_size]   #leaf nodes
    size_branch = size(branch_nodes,1)
    size_leaf = size(leaf_nodes,1)
    n = size(X,1)
    p = size(X,2)

    # classes=unique(Y)      #get a dataframe with the classes in the data
    K=size(classes,1)    #number of classes
    if tree_da == 1
        a, b, d = trans_params_d(candidate, p, size_branch)
    elseif tree_da == 2
        a, b, d = trans_params_a(candidate, p, size_branch)
    elseif tree_da == 3
        a, b, d = trans_params_az(candidate, p, size_branch)
    end

    # points assigned to each leaf node; zit = 𝟙 {𝐱i is in node t};
    # in this part, we track which 𝐱i arrives to t, ∀t ∈ leaf_nodes:   
    z = falses(n,size_leaf)                      
    for i in 1:n                                   
        t=1                     #always starts in the root node
        # print(" t: ", t)
        while true
            # println(X[i, :][a[:, t]])
            if d[t] && X[i, :][a[:, t]][1] < b[t]
                t = t*2
                if t>size_branch
                    break
                end
                # print(" t: ", t)
            else
                t = t*2 + 1
                if t>size_branch
                    break
                end
                # print(" t: ", t)
            end
        end
        # println(" t: ", t)
        z[i,t-size_branch]=true
    end
    octCost = 0.0
    for t in 1:size_leaf
        Nt = sum(z[:,t]) # Number of points at leaf node t
        Yt = Y[z[:,t]] 
        Nct = count(isequal(c[t]), Yt) # Number of points at leaf node t with label c[t]
        octCost = octCost + Nt - Nct # Cost of leaf node t
    end
    
    return octCost
end

end # end of module