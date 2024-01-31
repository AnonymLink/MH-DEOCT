using DataFrames, CSV
using LinearAlgebra, Random, Statistics
using TimerOutputs: @timeit, get_timer
# user defined modules:
if !("src/" in LOAD_PATH)
    push!(LOAD_PATH, "src/")
end    
using datasets, oct, local_search

const to = get_timer("Shared")

# Parameters
# julia test/test_local_search.jl 1 52 2 1 10 100
data_num_start = parse(Int, ARGS[1]) # number of the first dataset
data_num_end = parse(Int, ARGS[2]) # number of the last dataset
tree_depth = parse(Int, ARGS[3])
tree_size = 2 ^ (tree_depth + 1) - 1  
funcion = OCT
tree_da = 2
var_number = 2
Nmin = parse(Int, ARGS[4]) # minimum number of samples in a leaf; 0 - 1; 1 - ceil(Int64, 0.05*size(X,1)).
runs = parse(Int, ARGS[5]) # number of the last run
initials = parse(Int, ARGS[6]) # number of initial solutions
csv_flag = parse(Int, ARGS[7]) # 1 - save results to csv; 0 - don't save
if csv_flag == 1
    csv_path = ARGS[8] # path to save csv files
end
runs_start = parse(Int, ARGS[9]) # number of the first run

train_results=zeros(100, 10+2)
test_results=zeros(100, 10+2)
times = zeros(100, 10+2)
dir_path = "./data/"
for dataset in data_num_start:data_num_end
    println("######## Dataset: ", dataset, " ########")
    for run in runs_start:runs
        Random.seed!(run)
        println("## Run: ", run, " ##")
        train, val, test = loadDataset(dataset, run, dir_path)
        p = size(train, 2)-1
        X = vcat(train[:,1:p], val[:,1:p])
        Y = Int.(vcat(train[:,p+1], val[:,p+1]))
        X_test = test[:,1:p]
        Y_test = Int.(test[:,p+1])
        classes=sort(unique(Y))      #get a dataframe with the classes in the data
        K=size(classes,1)               #number of classes  
        tmp_Y = zeros(Int, size(Y,1))
        tmp_Y_test = zeros(Int, size(Y_test,1))
        for i in 1:K
            tmp_Y[findall(x->x==classes[i], Y)] .= i
            tmp_Y_test[findall(x->x==classes[i], Y_test)] .= i
        end
        Y = tmp_Y
        Y_test = tmp_Y_test
        classes=sort(unique(Y))      #get a dataframe with the classes in the data
        K=size(classes,1)               #number of classes  
        if run == runs_start
            println("dataset: ", dataset, " run: ", run, " n_train: ", size(X,1), " n: ", size(vcat(X, X_test), 1), " p: ", p, " K: ", size(unique(Y),1))
            println("tree_depth: ",tree_depth, ", tree_size: ",tree_size)
        end
        # local_searches(funcion::Function, tree_da, X, Y, classes, sortX, tree_size, Nmin=1)
        best_fitness = Inf
        xbest = zeros(Float64, floor(Int64,tree_size/2)*var_number)
        start = time()
        new_xbest = nothing
        new_finess = nothing
        for i in 1:initials            
            @timeit get_timer("Shared") "local_searches" begin
                if i == 1
                    new_xbest, new_finess =local_searches(funcion, tree_da, X, Y, classes, tree_size, Nmin, true)
                else
                    new_xbest, new_finess =local_searches(funcion, tree_da, X, Y, classes, tree_size, Nmin, false)
                end                
            end 
            if new_finess < best_fitness
                best_fitness = new_finess
                xbest = new_xbest
            end
        end
        times[dataset,run] = time() - start
        #it returns the out-of-sample accuracy per run per dataset:
        cost_train, c, Nmin_flag = funcion(tree_da, xbest, X, Y, classes, tree_size, Nmin, 4)
        println("c: ", c)
        train_results[dataset,run] = 100-(cost_train*100/size(X, 1))
        # OCT_test(c, tree_da, candidate, X, Y, classes, tree_size)
        test_results[dataset,run]=100-((OCT_test(c, tree_da, xbest, X_test, Y_test, classes, tree_size)*100)/size(X_test, 1)) 
        println("Run No: ", run, ", dataset No: ", dataset, ", %train_accuracy: ", train_results[dataset,run], ", %test_accuracy: ", test_results[dataset,run], ", Nmin_flag: ", Nmin_flag, ", time: ", times[dataset,run])

        if csv_flag == 1
            CSV.write(csv_path * "Alpha_0_LS_WS_fix_D" * string(tree_depth) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_init" * string(initials) * "_runs" * string(runs_start) * "_" * string(runs) * "_train.csv", DataFrame(train_results, :auto))
            CSV.write(csv_path * "Alpha_0_LS_WS_fix_D" * string(tree_depth) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_init" * string(initials) * "_runs" * string(runs_start) * "_" * string(runs) * "_test.csv", DataFrame(test_results, :auto))
            CSV.write(csv_path * "Alpha_0_LS_WS_fix_D" * string(tree_depth) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_init" * string(initials) * "_runs" * string(runs_start) * "_" * string(runs) * "_times.csv", DataFrame(times, :auto))
        end
    end # end for run
end # end for dataset

show(to)
# std and mean of the results
train_results[:, 10+1] = mean(train_results[:, 1:10], dims=2)
train_results[:, 10+2] = std(train_results[:, 1:10], dims=2)
test_results[:, 10+1] = mean(test_results[:, 1:10], dims=2)
test_results[:, 10+2] = std(test_results[:, 1:10], dims=2)
times[:, 10+1] = mean(times[:, 1:10], dims=2)
times[:, 10+2] = std(times[:, 1:10], dims=2)

# save results
if csv_flag == 1
    CSV.write(csv_path * "Alpha_0_LS_WS_fix_D" * string(tree_depth) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_init" * string(initials) * "_runs" * string(runs_start) * "_" * string(runs) * "_train.csv", DataFrame(train_results, :auto))
    CSV.write(csv_path * "Alpha_0_LS_WS_fix_D" * string(tree_depth) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_init" * string(initials) * "_runs" * string(runs_start) * "_" * string(runs) * "_test.csv", DataFrame(test_results, :auto))
    CSV.write(csv_path * "Alpha_0_LS_WS_fix_D" * string(tree_depth) * "_Data" * string(data_num_start) * "_" * string(data_num_end) * "_init" * string(initials) * "_runs" * string(runs_start) * "_" * string(runs) * "_times.csv", DataFrame(times, :auto))
end
