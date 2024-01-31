using DataFrames, CSV
using LinearAlgebra, Random, Statistics, StatsBase
using TimerOutputs: @timeit, get_timer
using Printf

using CUDA

# user defined modules:
if !("src_gpu/" in LOAD_PATH)
    push!(LOAD_PATH, "src_gpu/")
    push!(LOAD_PATH, "src_cart/")
end    
using DecisionTree_modified
using datasets_gpu, de_gpu, oct_gpu

# println cuda info
if CUDA.has_cuda()
    try
        println(CUDA.versioninfo())
    catch ex
        println("devices not available")
    end
else
    println("CUDA not available")
end

const to = get_timer("Shared")

# Parameters
data_num_start = parse(Int, ARGS[1]) # number of the first dataset 
data_num_end = parse(Int, ARGS[2]) # number of the last dataset 
tree_depth = parse(Int, ARGS[3]) # 
tree_size = 2 ^ (tree_depth + 1) - 1  
tree = ARGS[4] # abandoned parameter - not used
if tree == "OCT_a"
    tree_da = 2
    var_number = 2
    tree_da = 1
    var_number = 3
elseif tree == "OCT_az"
    tree_da = 3
    var_number = 2
end
NminIdx = parse(Int, ARGS[5]) # minimum number of samples in a leaf; 0 - 1; 1 - ceil(Int64, 0.05*size(X,1)).
runs = parse(Int, ARGS[6]) # number of the last run
P = parse(Int, ARGS[7]) # moving horizon depth
csv_flag = parse(Int, ARGS[8]) # 0 - no write csv, 1 - write csv
if csv_flag == 1
    csv_path = ARGS[9] # path to csv file
end
polish_flag = parse(Int, ARGS[10]) # abandoned parameter - not used
N_population = parse(Int32, ARGS[11]) # population size in DE - not used
DE_iters = parse(Int, ARGS[12]) # number of iterations in DE - not used
ws_flag = parse(Int, ARGS[13]) # 0 - no warm start, 1 - warm start - not used 
ws_flag = ws_flag == 1 ? true : false
init_flag = parse(Int, ARGS[14]) # 0 - xbest from de, 1 - CART, 2 - nothing - not used
runs_start = parse(Int, ARGS[15]) # number of the first run

# Data
cart_times = zeros(100, 10+2)
cartTrainAccur = zeros(100, 10+2)
cartTestAccur = zeros(100, 10+2)
val_results = zeros(100*10, 2000) # alphas, val_accs
best_alphas = zeros(100, 10+2)
dir_path = "./data/"

# GPU Initializing
@timeit get_timer("Shared") "gpu_init" kernels, threadss = gpu_init()

# Main loop
for dataset in data_num_start:data_num_end
    println("############ Dataset: ", dataset, " ############")
    for run in runs_start:runs
        @timeit get_timer("Shared") "gc" GC.gc()
        Random.seed!(run)
        println("##############################")
        println("########### Run: ", run, " ###########")
        train, val, test = loadDataset(dataset, run, dir_path)
        p = size(train, 2)-1
        L_hat = maximum(counts(Int.(train[:,p+1])))/length(Int.(train[:,p+1]))
        Y = Int.(vcat(train[:,p+1], val[:,p+1], test[:,p+1]))
        X_train = train[:,1:p]
        Y_train = Int.(train[:,p+1])
        X_val = val[:,1:p]
        Y_val = Int.(val[:,p+1])
        X_test = test[:,1:p]
        Y_test = Int.(test[:,p+1])
        # println("size_X: ", size(X_train), " size_Y: ", size(Y_train))
        classes=sort(unique(Y))      #get a dataframe with the classes in the data
        class_labels=size(classes,1)               #number of classes  
        tmp_Y_train = zeros(Int, size(Y_train,1))
        tmp_Y_val = zeros(Int, size(Y_val,1))
        tmp_Y_test = zeros(Int, size(Y_test,1))
        for i in 1:class_labels
            tmp_Y_train[findall(x->x==classes[i], Y_train)] .= i
            tmp_Y_val[findall(x->x==classes[i], Y_val)] .= i
            tmp_Y_test[findall(x->x==classes[i], Y_test)] .= i
        end
        Y_train = tmp_Y_train
        Y_val = tmp_Y_val
        Y_test = tmp_Y_test
        # println("size_X: ", size(X_train), " size_Y: ", size(Y_train))
        classes=sort(unique(Y))      #get a dataframe with the classes in the data
        class_labels=size(classes,1)               #number of classes  
        if run == 1
            println("dataset: ", dataset, " run: ", run, " n_train: ", size(X_train,1), " n: ", size(X_train, 1) + size(X_val, 1) + size(X_test, 1), " p: ", p, " class_labels: ", class_labels)
            println("tree: ", tree, ", tree_depth: ",tree_depth, ", tree_size: ",tree_size)
        end

        if NminIdx==0
            Nmin = 1
        elseif NminIdx==1
            Nmin = ceil(Int64, 0.05*size(X,1))
        end
        X = X_train
        Y = Y_train
        # println("size_X: ", size(X), " size_Y: ", size(Y))
        Y_K = StatsBase.indicatormat(Y, class_labels)
        X_d = CuArray(Float32.(X))

        alphas = 0:0.05:2
        best_alpha = alphas[1]
        best_val_cost = Inf
        best_xbest = zeros(1, tree_size)
        alpha_index = 1
        println("######### alpha tuning #########")
        println("######### alpha tuning #########")
        @timeit get_timer("Shared") "alpha_tuning" begin
        for alpha in alphas
            @timeit get_timer("Shared") "calculations" begin
            model = DecisionTree_modified.build_tree(Y, X, 0, tree_depth, 1)
            model = DecisionTree_modified.prune_tree(model, alpha)
            preds_train = apply_tree(model, X)
            cost_train = sum(preds_train .!= Y)/length(Y)
            preds_val = apply_tree(model, X_val)
            val_cost = sum(preds_val .!= Y_val)/length(Y_val)
            println("Run No: ", run, ", dataset No: ", dataset, ", alpha: ", alpha, ", val_cost: ", val_cost, ", train_cost: ", cost_train, ", best_val_cost: ", best_val_cost)
            if val_cost < best_val_cost
                best_val_cost = val_cost
                best_alpha = alpha
                best_xbest = model
                println("****** Updated Best: best_alpha: ", best_alpha, ", best_val_cost: ", best_val_cost)
            end
            val_results[(dataset-1)*runs+run, alpha_index] = alpha
            val_results[(dataset-1)*runs+run, alpha_index + length(alphas)] = val_cost
            alpha_index += 1
            end # end for @timeit get_timer("Shared") "calculations"
        end # end for alpha
    end # end for @timeit get_timer("Shared") "alpha_tuning"

        # retrain after validating
        println("######### retrain #########")
        println("######### retrain #########")
        @timeit get_timer("Shared") "retrain" begin
        best_alphas[dataset, run] = best_alpha
        alpha = best_alpha
        Random.seed!(run)
        X = vcat(X_train, X_val)
        Y = Int.(vcat(Y_train, Y_val))

        @timeit get_timer("Shared") "calculations" begin
        start = time()
        model = DecisionTree_modified.build_tree(Y, X, 0, tree_depth, 1)
        model = DecisionTree_modified.prune_tree(model, alpha)
        preds_train = apply_tree(model, X)
        cost_train = sum(preds_train .!= Y)/length(Y)
        preds_train1 = apply_tree(best_xbest, X)
        cost_train1 = sum(preds_train1 .!= Y)/length(Y)
        preds_test = apply_tree(model, X_test)
        acc_test = 100-((sum(preds_test .!= Y_test)*100)/size(X_test, 1))
        preds_test1 = apply_tree(best_xbest, X_test)
        acc_test1 = 100-((sum(preds_test1 .!= Y_test)*100)/size(X_test, 1))

        if acc_test < acc_test1
            xbest = best_xbest
            cost_train = cost_train1
            acc_test = acc_test1
        end
        cartTrainAccur[dataset,run] = 100-(cost_train*100)
        cartTestAccur[dataset,run] = acc_test
        cart_times[dataset,run] = time() - start
        println("Layer: Run No: ", run, ", dataset No: ", dataset, ", %train_accuracy: ", cartTrainAccur[dataset,run], ", %test_accuracy: ", cartTestAccur[dataset,run], " time: ", cart_times[dataset,run])
    end # end for @timeit get_timer("Shared") "calculations"
    end # end for @timeit get_timer("Shared") "retrain"
        # save results at the end of each run
        if csv_flag == 1
            # save the results
            cart_times_csv=DataFrame(cart_times, :auto)
            CART_train_accur_dffinal_csv=DataFrame(cartTrainAccur, :auto)
            CART_test_accur_dffinal_csv=DataFrame(cartTestAccur, :auto)
            val_results_csv=DataFrame(val_results, :auto)
            best_alphas_csv=DataFrame(best_alphas, :auto)

            if ws_flag
                CSV.write(csv_path * "CART_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) *  "_train_CART.csv", CART_train_accur_dffinal_csv)
                CSV.write(csv_path * "CART_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*   "_runs" * string(runs_start) * "_" * string(runs) * "_test_CART.csv", CART_test_accur_dffinal_csv)
                CSV.write(csv_path * "CART_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) *  "_cart_times.csv", cart_times_csv)
            end
            # val_results_csv
            CSV.write(csv_path * "CART_" * string(N_population) * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)* "_runs" * string(runs_start) * "_" * string(runs) * "_val_results.csv", val_results_csv)
            CSV.write(csv_path * "CART_" * string(N_population) * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)* "_runs" * string(runs_start) * "_" * string(runs) * "_best_alphas.csv", best_alphas_csv)
        end # end for csv_flag
    end # end for run
end # end for dataset

print('\n')
show(to)
print('\n')

# CART and RF 
cartTrainAccur[:,10+2] = std(cartTrainAccur[:, 1:10], dims=2)
cartTrainAccur[:, 10+1] = mean(cartTrainAccur[:, 1:10], dims=2)
cartTestAccur[:,10+2] = std(cartTestAccur[:, 1:10], dims=2)
cartTestAccur[:, 10+1] = mean(cartTestAccur[:, 1:10], dims=2)
cart_times[:,10+2] = std(cart_times[:, 1:10], dims=2)
cart_times[:, 10+1] = mean(cart_times[:, 1:10], dims=2)

if csv_flag == 1
    # save the results
    cart_times=DataFrame(cart_times, :auto)
    CART_train_accur_dffinal=DataFrame(cartTrainAccur, :auto)
    CART_test_accur_dffinal=DataFrame(cartTestAccur, :auto)
    val_results=DataFrame(val_results, :auto)
    best_alphas=DataFrame(best_alphas, :auto)

    if ws_flag
        CSV.write(csv_path * "CART_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) *  "_train_CART.csv", CART_train_accur_dffinal)
        CSV.write(csv_path * "CART_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) *  "_test_CART.csv", CART_test_accur_dffinal)
        CSV.write(csv_path * "CART_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) *  "_cart_times.csv", cart_times)
    end
    # val_results_csv
    CSV.write(csv_path * "CART_" * string(N_population) * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)* "_runs" * string(runs_start) * "_" * string(runs) * "_val_results.csv", val_results)
    CSV.write(csv_path * "CART_" * string(N_population) * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)* "_runs" * string(runs_start) * "_" * string(runs) * "_best_alphas.csv", best_alphas)
end

