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
tree_depth = parse(Int, ARGS[3]) # tree depth
tree_size = 2 ^ (tree_depth + 1) - 1  # 
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
P = parse(Int, ARGS[7]) # Moving horizon depth
csv_flag = parse(Int, ARGS[8]) # 0 - no write csv, 1 - write csv
if csv_flag == 1
    csv_path = ARGS[9] # path to csv files
end
polish_flag = parse(Int, ARGS[10]) # abandoned parameter - not used
N_population = parse(Int32, ARGS[11]) # size of population in DE
DE_iters = parse(Int, ARGS[12]) # maximum generation in DE
ws_flag = parse(Int, ARGS[13]) # 0 - no warm start, 1 - warm start
ws_flag = ws_flag == 1 ? true : false
init_flag = parse(Int, ARGS[14]) # 0 - xbest from de, 1 - CART, 2 - nothing
runs_start = parse(Int, ARGS[15]) # number of the first run

# Data
train_results_DE = zeros(100, 10+2)
test_results_DE = zeros(100, 10+2)
train_results=zeros(100, 10+2)
test_results=zeros(100, 10+2)
UB_results=zeros(100, 10+2)
train_unique_ratio = zeros(100, 10+2)
times = zeros(100, 10+2)
cart_times = zeros(100, 10+2)
de_times = zeros(100, 10+2)
cartTrainAccur = zeros(100, 10+2)
cartTestAccur = zeros(100, 10+2)
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
        X = vcat(train[:,1:p], val[:,1:p])
        Y = Int.(vcat(train[:,p+1], val[:,p+1]))
        X_test = test[:,1:p]
        Y_test = Int.(test[:,p+1])
        classes=sort(unique(Y))      #get a dataframe with the classes in the data
        class_labels=size(classes,1)               #number of classes  
        tmp_Y = zeros(Int, size(Y,1))
        tmp_Y_test = zeros(Int, size(Y_test,1))
        for i in 1:class_labels # sort the number of classes to avoid labels like 1,2,5,6 to 1,2,3,4
            tmp_Y[findall(x->x==classes[i], Y)] .= i
            tmp_Y_test[findall(x->x==classes[i], Y_test)] .= i
        end
        Y = tmp_Y
        Y_test = tmp_Y_test
        classes=sort(unique(Y))      #get a dataframe with the classes in the data
        class_labels=size(classes,1)               #number of classes  
        if run == 1
            println("dataset: ", dataset, " run: ", run, " n_train: ", size(X,1), " n: ", size(vcat(X, X_test), 1), " p: ", p, " class_labels: ", size(unique(Y),1))
            println("tree: ", tree, ", tree_depth: ",tree_depth, ", tree_size: ",tree_size)
        end

        if NminIdx==0
            Nmin = 1
        elseif NminIdx==1
            Nmin = ceil(Int64, 0.05*size(X,1))
        end
        Y_K = StatsBase.indicatormat(Y, class_labels)
        X_d = CuArray(Float32.(X))
        fun_args = args_pre(class_labels, tree_da, X_d, X, Y, tree_size, Nmin, kernels, threadss, N_population, 0.0) # prepare parameters for DE and MH
        original_splits = deepcopy(fun_args[13])
        if dataset == data_num_start && run == 1 # Initializing
            @timeit get_timer("Shared") "deb1b-init" xbest, cartModel, DT_warmstart,  = DEb1b_warmStart(fun_args, tree_da, 5, X, Y, tree_size, var_number, Nmin, N_population, 0.5, 0.1, 1)
        end

        @timeit get_timer("Shared") "calculations" begin
        start = time()
        if init_flag !=2
            @timeit get_timer("Shared") "DEb1b" xbest, cartModel, DT_warmstart, ratio, cart_time = DEb1b_warmStart(fun_args, tree_da, DE_iters, X, Y, tree_size, var_number, Nmin, N_population, 0.5, 0.1, 1, nothing, nothing, ws_flag, init_flag)
            train_unique_ratio[dataset,run] = ratio
            de_times[dataset,run] = time() - start
            cart_times[dataset,run] = cart_time
            #it returns the out-of-sample accuracy per run per dataset:
            cost_train, c, Nmin_flag = OCT(tree_da, xbest, X, Y_K, classes, tree_size, Nmin, 4, 0.05, original_splits)
            train_results_DE[dataset,run] = 100-(cost_train*100/size(X, 1))
            test_results_DE[dataset,run] = 100-((OCT_test(c, tree_da, xbest, X_test, Y_test, classes, tree_size, original_splits)*100)/size(X_test, 1))    
            println("DEb1b: Run No: ", run, ", dataset No: ", dataset, ", %train_accuracy: ", train_results_DE[dataset,run], ", %test_accuracy: ", test_results_DE[dataset,run], ", ratio: ", ratio, ", Nmin_flag: ", Nmin_flag, ", DE_time: ", de_times[dataset,run])     
        end

        # calculate the CART and RF (for warm start) accuracy and the fitness
        if ws_flag
            preds_train = DecisionTree_modified.apply_tree(cartModel, X)
            cm_train = DecisionTree_modified.confusion_matrix(Y, preds_train)
            preds_test  = DecisionTree_modified.apply_tree(cartModel, X_test)
            cm_test = DecisionTree_modified.confusion_matrix(Y_test, preds_test)
            cartTrainAccur[dataset,run] = cm_train.accuracy*100
            cartTestAccur[dataset,run] = cm_test.accuracy*100
            println("CART: Run No: ", run, ", dataset No: ", dataset, ", %train_accuracy: ", cartTrainAccur[dataset,run], ", %test_accuracy: ", cartTestAccur[dataset,run], ", cart_time: ", cart_time)
        end
        
        mode = 1
        if init_flag == 0
            println("********* DE_LA *********")
            @timeit get_timer("Shared") "layer_by_layer" xbest=layer_by_layer_original_warmStart(fun_args, tree_da, P, DE_iters, X_d, X, Y, Y_K, tree_size, var_number, Nmin, N_population, 0.5, 0.1, 1, xbest)
        elseif init_flag == 1
            println("********* CART_LA *********")
            start = time()
            @timeit get_timer("Shared") "layer_by_layer" xbest=layer_by_layer_original_warmStart(fun_args, tree_da, P, DE_iters, X_d, X, Y, Y_K, tree_size, var_number, Nmin, N_population, 0.5, 0.1, 1, DT_warmstart)
        elseif init_flag == 2
            println("********* NoWS_LA *********")
            start = time()
            @timeit get_timer("Shared") "layer_by_layer" xbest=layer_by_layer_original_warmStart(fun_args, tree_da, P, DE_iters, X_d, X, Y, Y_K, tree_size, var_number, Nmin, N_population, 0.5, 0.1, 1, nothing)
        elseif init_flag == 10
            println("********* DE-LA-fullnocartws *********")
            @timeit get_timer("Shared") "layer_by_layer" xbest=layer_by_layer_original_warmStart(fun_args, tree_da, P, DE_iters, X_d, X, Y, Y_K, tree_size, var_number, Nmin, N_population, 0.5, 0.1, 1, xbest, ws_flag)
        elseif init_flag == 12
            println("********* LA-fullnocartws *********")
            start = time()
            @timeit get_timer("Shared") "layer_by_layer" xbest=layer_by_layer_original_warmStart(fun_args, tree_da, P, DE_iters, X_d, X, Y, Y_K, tree_size, var_number, Nmin, N_population, 0.5, 0.1, 1, nothing, ws_flag)
        elseif init_flag == 110
            println("********* DE-LA_only *********")
            @timeit get_timer("Shared") "layer_by_layer" xbest=layer_by_layer_original_warmStart(fun_args, tree_da, P, DE_iters, X_d, X, Y, Y_K, tree_size, var_number, Nmin, N_population, 0.5, 0.1, 1, xbest, false)
        end
        #it returns the out-of-sample accuracy per run per dataset:        
        cost_train, c, Nmin_flag = OCT(tree_da, xbest, X, Y_K, classes, tree_size, Nmin, 4, 0.05, original_splits)
        train_results[dataset,run] = 100-(cost_train*100/size(X, 1))
        UB = get_UB(cost_train, L_hat, xbest, tree_da, tree_size, p, 0.05, original_splits)
        UB_results[dataset,run] = UB
        test_results[dataset,run]=100-((OCT_test(c, tree_da, xbest, X_test, Y_test, classes, tree_size, original_splits)*100)/size(X_test, 1)) 
        times[dataset,run] = time() - start
        if init_flag == 1
            times[dataset,run] = times[dataset,run] + cart_times[dataset,run]
        end
        println("Layer: Run No: ", run, ", dataset No: ", dataset, ", %train_accuracy: ", train_results[dataset,run], ", %test_accuracy: ", test_results[dataset,run], ", UB: ", UB_results[dataset,run], ", Nmin_flag: ", Nmin_flag, ", time: ", times[dataset,run])
        
        println("Layer: Run No: ", run, ", dataset No: ", dataset, ", xbest: ", xbest)
        end # end for @timeit get_timer("Shared") "calculations"
        # save results at the end of each run
        if csv_flag == 1
            # save the results
            train_results_csv=DataFrame(train_results, :auto)
            test_results_csv=DataFrame(test_results, :auto)
            times_csv = DataFrame(times, :auto)
            train_results_DE_csv=DataFrame(train_results_DE, :auto)
            test_results_DE_csv=DataFrame(test_results_DE, :auto)
            train_unique_ratio_csv=DataFrame(train_unique_ratio, :auto)
            cart_times_csv=DataFrame(cart_times, :auto)
            de_times_csv=DataFrame(de_times, :auto)
            UB_results_csv=DataFrame(UB_results, :auto)
            CART_train_accur_dffinal_csv=DataFrame(cartTrainAccur, :auto)
            CART_test_accur_dffinal_csv=DataFrame(cartTestAccur, :auto)
        
            if ws_flag
                CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) *  "_train_CART.csv", CART_train_accur_dffinal_csv)
                CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*   "_runs" * string(runs_start) * "_" * string(runs) * "_test_CART.csv", CART_test_accur_dffinal_csv)
                CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) *  "_cart_times.csv", cart_times_csv)
            end
            CSV.write(csv_path * "LO_WS_Np_" * string(N_population) * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)* "_runs" * string(runs_start) * "_" * string(runs) * "_train_lbl.csv", train_results_csv)
            CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) * "_test_lbl.csv", test_results_csv)
            CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*   "_runs" * string(runs_start) * "_" * string(runs) * "_times.csv", times_csv)
            CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*   "_runs" * string(runs_start) * "_" * string(runs) * "_unique_ratio.csv", train_unique_ratio_csv)
            CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) * "_train_DE.csv", train_results_DE_csv)
            CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*   "_runs" * string(runs_start) * "_" * string(runs) * "_test_DE.csv", test_results_DE_csv)
            CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) *  "_UB.csv", UB_results_csv)
            CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) *  "_de_times.csv", de_times_csv)
        end
    end # end for run
end # end for dataset

print('\n')
show(to)
print('\n')

# std and mean of the results
train_results[:, 10+2] = std(train_results[:, 1:10], dims=2)
train_results[:, 10+1] = mean(train_results[:, 1:10], dims=2)
test_results[:, 10+2] = std(test_results[:, 1:10], dims=2)
test_results[:, 10+1] = mean(test_results[:, 1:10], dims=2)
times[:, 10+2] = std(times[:, 1:10], dims=2)
times[:, 10+1] = mean(times[:, 1:10], dims=2)
train_results_DE[:, 10+2] = std(train_results_DE[:, 1:10], dims=2)
train_results_DE[:, 10+1] = mean(train_results_DE[:, 1:10], dims=2)
test_results_DE[:, 10+2] = std(test_results_DE[:, 1:10], dims=2)
test_results_DE[:, 10+1] = mean(test_results_DE[:, 1:10], dims=2)
train_unique_ratio[:, 10+2] = std(train_unique_ratio[:, 1:10], dims=2)
train_unique_ratio[:, 10+1] = mean(train_unique_ratio[:, 1:10], dims=2)

# CART and RF 
cartTrainAccur[:,10+2] = std(cartTrainAccur[:, 1:10], dims=2)
cartTrainAccur[:, 10+1] = mean(cartTrainAccur[:, 1:10], dims=2)
cartTestAccur[:,10+2] = std(cartTestAccur[:, 1:10], dims=2)
cartTestAccur[:, 10+1] = mean(cartTestAccur[:, 1:10], dims=2)
cart_times[:,10+2] = std(cart_times[:, 1:10], dims=2)
cart_times[:, 10+1] = mean(cart_times[:, 1:10], dims=2)
de_times[:,10+2] = std(de_times[:, 1:10], dims=2)
de_times[:, 10+1] = mean(de_times[:, 1:10], dims=2)
UB_results[:,10+2] = std(UB_results[:, 1:10], dims=2)
UB_results[:, 10+1] = mean(UB_results[:, 1:10], dims=2)

if csv_flag == 1
    # save the results
    train_results=DataFrame(train_results, :auto)
    test_results=DataFrame(test_results, :auto)
    times = DataFrame(times, :auto)
    train_results_DE=DataFrame(train_results_DE, :auto)
    test_results_DE=DataFrame(test_results_DE, :auto)
    train_unique_ratio=DataFrame(train_unique_ratio, :auto)
    cart_times=DataFrame(cart_times, :auto)
    de_times=DataFrame(de_times, :auto)
    UB_results=DataFrame(UB_results, :auto)
    CART_train_accur_dffinal=DataFrame(cartTrainAccur, :auto)
    CART_test_accur_dffinal=DataFrame(cartTestAccur, :auto)

    if ws_flag
        CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) *  "_train_CART.csv", CART_train_accur_dffinal)
        CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) *  "_test_CART.csv", CART_test_accur_dffinal)
        CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) *  "_cart_times.csv", cart_times)
    end
    CSV.write(csv_path * "LO_WS_Np_" * string(N_population) * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)*  "_runs" * string(runs_start) * "_" * string(runs) *  "_train_lbl.csv", train_results)
    CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)* "_runs" * string(runs_start) * "_" * string(runs) *  "_test_lbl.csv", test_results)
    CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)* "_runs" * string(runs_start) * "_" * string(runs) *   "_times.csv", times)
    CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)* "_runs" * string(runs_start) * "_" * string(runs) *   "_unique_ratio.csv", train_unique_ratio)
    CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)* "_runs" * string(runs_start) * "_" * string(runs) *  "_train_DE.csv", train_results_DE)
    CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)* "_runs" * string(runs_start) * "_" * string(runs) *   "_test_DE.csv", test_results_DE)
    CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)* "_runs" * string(runs_start) * "_" * string(runs) *   "_UB.csv", UB_results)
    CSV.write(csv_path * "LO_WS_Np_" * string(N_population)  * tree * "_D" * string(tree_depth) * "_P" * string(P) * "_DE" * string(DE_iters) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_ws_flag" * string(ws_flag) * "_init"* string(init_flag)* "_runs" * string(runs_start) * "_" * string(runs) *   "_de_times.csv", de_times)
end