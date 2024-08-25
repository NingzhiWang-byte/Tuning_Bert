%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is to replicate the results of Random Forest model          %
% Our results are generated using Matlab R2020b on Windows 10           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
auc_values = [];
ncdg_values = [];
sensitivity_values = [];
precision_values = [];
Top_N = [];
Test_year = [];

% Initialize arrays to store results
results = table(Test_year, Top_N, auc_values, ncdg_values, sensitivity_values, precision_values);

diary("results_random_forest.txt");
for year_test = 2014:2017
    rng(0,'twister'); % fix random seed for reproducing the results
    % read training data
    fprintf('==> Running Random Forest (training period: %d-%d, testing period: %d)...\n',2011,year_test-1,year_test);
    data_train = data_reader('env.csv','data_default',2011,year_test-2);
    y_train = data_train.labels;
    X_train = data_train.features;
    paaer_train = data_train.paaers;
    
    % read testing data
    data_test = data_reader('env.csv','data_default',year_test-1,year_test);
    y_test = data_test.labels;
    X_test = data_test.features;
    paaer_test = unique(data_test.paaers(data_test.labels~=0));
    
    % handle serial frauds using PAAER
    y_train(ismember(paaer_train,paaer_test)) = 0;

    % train model
    t1 = tic;
    t = templateTree('MinLeafSize',5); % base model
    % Create a Random Forest model with 500 trees
    rf_model = TreeBagger(500, X_train, y_train, 'Method', 'classification', 'OOBPrediction', 'on');
    t_train = toc(t1);
    
    % test model
    t2 = tic;
    [label_predict, scores] = predict(rf_model, X_test);
    dec_values = scores(:,2); % Extract the scores for the positive class
    t_test = toc(t2);
    
    % Convert label_true, label_predict, and dec_values to compatible data type
    label_true = double(y_test);
    label_predict = cell2mat(label_predict);
    dec_values = double(dec_values);
    
    % print performance results
    fprintf('Training time: %g seconds | Testing time %g seconds \n', t_train, t_test);
    for topN = [0.01, 0.02, 0.03, 0.04, 0.05]
        metrics = evaluate(y_test,label_predict,dec_values,topN);
        fprintf('Performance (top%d%% as cut-off thresh): \n',topN*100);
        fprintf('AUC: %.4f \n', metrics.auc);
        fprintf('NCDG@k: %.4f \n', metrics.ndcg_at_k);
        fprintf('Sensitivity: %.2f%% \n', metrics.sensitivity_topk*100);
        fprintf('Precision: %.2f%% \n', metrics.precision_topk*100);
        
        new_row = {year_test, topN, metrics.auc, metrics.ndcg_at_k, metrics.sensitivity_topk, metrics.precision_topk};
        results = [results; new_row];
    end
end
diary off;

% Write results to a CSV file
writetable(results,'C:\\Users\\sz_wh\\Desktop\\【data】\\【毕业论文用】fraud论文复刻\\GreenwashingDetection\\results\\test2\\RandomForest_test_env3.csv');
