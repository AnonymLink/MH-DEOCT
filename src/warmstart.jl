module warmstart
using DecisionTree
using StatsBase
using Random

export warm_start_DT, warm_start_DT_params

# warm start of Decision Tree (CART)
function warm_start_DT(X, y,tree_da, var_number, Nmin, D = 4, prune_val=0.0)
    n,p = size(X)
    Tb = 2^D-1
    T = 2^(D+1)-1
    # Random.seed!(1)
    # cart_model = DecisionTree.build_tree(y, X', 0, D)
    cart_model = DecisionTree.build_tree(y, X, 0, D, Nmin)
    # println(typeof(cart_model))
    # DecisionTree.print_tree(cart_model)
    a,b,c,d = warm_start_DT_params(zeros(p,Tb), zeros(Tb), zeros(T), zeros(Tb), 1, cart_model.node, 2^D:T)

    # change binary a to 1-d float number, which is used in Differential evolution
    if tree_da !=3
        aFloat =  zeros(Float64, Tb)
        for j in 1:Tb
            if iszero(a[:,j])
                aFloat[j] = 0.0
            else
                idx_one = findall(!iszero, a[:,j])
                oneToFloat =  (idx_one[1]-1)/p .+ rand(Float64,1)*((idx_one[1])/p-(idx_one[1]-1)/p)
                aFloat[j] = oneToFloat[1]
            end
        end
    else
        aFloat =  zeros(Float64, Tb)
        for j in 1:Tb
            if iszero(a[:,j])
                aFloat[j] = ((p)/(p+1) .+ rand(Float64,1)*( 1-(p)/(p+1)))[1]
            else
                idx_one = findall(!iszero, a[:,j])
                oneToFloat =  (idx_one[1]-1)/(p+1) .+ rand(Float64,1)*((idx_one[1])/(p+1)-(idx_one[1]-1)/(p+1))
                aFloat[j] = oneToFloat[1]
            end
        end

    end 

    # return Tree(aFloat,d,b)
    if var_number == 3
        DT_adb = vcat(aFloat,d,b)                         # vector (9,)
        DT_adb_mat =  reshape(DT_adb,1,length(DT_adb))    #（1， 9）
        
    else var_number == 2 
        DT_adb = vcat(aFloat,b)                         # vector (9,)
        DT_adb_mat =  reshape(DT_adb,1,length(DT_adb))    #（1， 9）
    end

     # calculate cart accuracy 
     preds_train = DecisionTree.apply_tree(cart_model, X)
     cm_train = DecisionTree.confusion_matrix(y, preds_train)
     println("CART_warmstart training accuracy  is: ", cm_train.accuracy)

    return DT_adb_mat, cart_model
end

function warm_start_DT_params(a,b,c,d,t,node,Tl)
    if node isa Leaf
        t_leaf = t
        while !(t_leaf in Tl)
            t_leaf = 2*t_leaf+1
        end
        #println("$t_leaf from $t")
        c[t_leaf] = node.majority
    else
        a[node.featid, t] = 1
        b[t] = node.featval
        d[t] = 1
        a,b,c,d = warm_start_DT_params(a,b,c,d,2*t, node.left, Tl)
        a,b,c,d = warm_start_DT_params(a,b,c,d,2*t+1, node.right, Tl)
    end
    return a,b,c,d
end

end 