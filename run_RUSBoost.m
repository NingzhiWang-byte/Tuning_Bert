%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is to replicate the results of RUSBoost model               %
% Our results are generated using Matlab R2020b on Windows 10           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%file_path = 'C:\\Users\\sz_wh\\Desktop\\【data】\\【毕业论文用-用行政处罚建立一个新的Y】\\data\\results\\';
file_path = 'C:\\Users\\sz_wh\\Desktop\\【data】\\【毕业论文用-用行政处罚建立一个新的Y】\\data\\results_southweekend\\';
%%%%% 2022 %%%%%
file_input = 'env_winsored.csv';
file_output = 'RUSBoost_2022_env.csv';
print_test_rusboost_2022(file_input,file_output,file_path);
file_input = 'noenv_winsored.csv';
file_output = 'RUSBoost_2022_noenv.csv';
print_test_rusboost_2022(file_input,file_output,file_path);
file_input = 'financial_winsored.csv';
file_output = 'RUSBoost_2022_financial.csv';
print_test_rusboost_2022(file_input,file_output,file_path);


file_input = 'env_winsored.csv';
file_output = 'RUSBoost_test_env3.csv';
print_test_rusboost(file_input,file_output,file_path);
file_input = 'noenv_winsored.csv';
file_output = 'RUSBoost_test_noenv3.csv';
print_test_rusboost(file_input,file_output,file_path);
file_input = 'financial_winsored.csv';
file_output = 'RUSBoost_test_financial3.csv';
print_test_rusboost(file_input,file_output,file_path);



%%%%% baseline %%%%%%

file_input = 'env_winsored.csv';
file_output = 'Rusboost_baseline_env_nopaar.csv';
nopaar = 0;
print_rustboost_baseline(file_input,file_output,nopaar,file_path)
file_input = 'noenv_winsored.csv';
file_output = 'Rusboost_baseline_noenv_nopaar.csv';
nopaar = 0;
print_rustboost_baseline(file_input,file_output,nopaar,file_path)
file_input = 'financial_winsored.csv';
file_output = 'Rusboost_baseline_financial_nopaar.csv';
nopaar = 0;
print_rustboost_baseline(file_input,file_output,nopaar,file_path)


%file_input = 'envonly_winsored.csv';
%file_output = 'Rusboost_baseline_envonly.csv';
%nopaar = 1;
%print_rustboost_baseline(file_input,file_output,nopaar)
%file_input = 'noenvonly_winsored.csv';
%file_output = 'Rusboost_baseline_noenvonly.csv';
%nopaar = 1;
%print_rustboost_baseline(file_input,file_output,nopaar)
%file_input = 'csmaronly_winsored.csv';
%file_output = 'Rusboost_baseline_csmaronly.csv';
%nopaar = 1;
%print_rustboost_baseline(file_input,file_output,nopaar)

function result = print_test_rusboost(file_input,file_output,file_path)
    iters = 300;
    lrate = 0.1;
    small = 1;
    large = 1;

    % financial 
    auc_values = [];
    fscore_values = [];
    ncdg_values = [];
    sensitivity_values = [];
    precision_values = [];
    Top_N = [];
    Test_year = [];

    % Initialize arrays to store results
    results = table(Test_year, Top_N, auc_values, ncdg_values, sensitivity_values, precision_values,fscore_values);



    diary("results_rusboost.txt");
    for year_test = 2014:2016
        rng(0,'twister'); % fix random seed for reproducing the results
        % read training data
        fprintf('==> Running RUSBoost (training period: %d-%d, testing period: %d-%d)...\n',2010,year_test-2,year_test-1,year_test);
        data_train = data_reader(file_input,'data_default',2010,year_test-2);
        y_train = data_train.labels;
        X_train = data_train.features;
        paaer_train = data_train.paaers;

        % read testing data
        data_test = data_reader(file_input,'data_default',year_test-1,year_test);
        y_test = data_test.labels;
        X_test = data_test.features;
        paaer_test = unique(data_test.paaers(data_test.labels~=0));

        % handle serial frauds using PAAER
        y_train(ismember(paaer_train,paaer_test)) = 0;

        % train model
        t1 = tic;
        t = templateTree('MinLeafSize',6); % base model
        rusboost = fitensemble(X_train,y_train,'RUSBoost',iters,t,'LearnRate',lrate,'RatioToSmallest',[small large]);
        t_train = toc(t1);

        % test model
        t2 = tic;
        [label_predict,dec_values] = predict(rusboost,X_test);
        dec_values = dec_values(:,2);
        t_test = toc(t2);


        % print performance results
        fprintf('Training time: %g seconds | Testing time %g seconds \n', t_train, t_test);
        for topN = [0.01, 0.02, 0.03, 0.04, 0.05]
            metrics = evaluate(y_test,label_predict,dec_values,topN);
            fprintf('Performance (top%d%% as cut-off thresh): \n',topN*100);
            fprintf('AUC: %.4f \n', metrics.auc);
            fprintf('F-score: %.4f \n', metrics.fscore_topk);
            fprintf('NCDG@k: %.4f \n', metrics.ndcg_at_k);
            fprintf('Sensitivity: %.2f%% \n', metrics.sensitivity_topk*100);
            fprintf('Precision: %.2f%% \n', metrics.precision_topk*100);
            new_row = {year_test, topN, metrics.auc,  metrics.ndcg_at_k, metrics.sensitivity_topk, metrics.precision_topk,metrics.fscore_topk};
            results = [results; new_row];
        end
    end
    diary off;
    output_file = strcat(file_path,file_output);
    % Write results to a CSV file
    writetable(results,output_file);

end

function result = print_rustboost_baseline(file_input,file_output,nopaar,file_path)
    iters = 300;
    lrate = 0.1;
    small = 1;
    large = 1;
    %%%%%%%%%%%%%%%% envonly
    auc_values = [];
    fscore_values = [];
    ncdg_values = [];
    sensitivity_values = [];
    precision_values = [];
    Top_N = [];
    Test_year = [];


    % Initialize arrays to store results
    results = table(Test_year, Top_N, auc_values, ncdg_values, sensitivity_values, precision_values, fscore_values);


    diary("results_rusboost.txt");
    year_test = 2016;
    rng(0,'twister'); % fix random seed for reproducing the results
    % read training data
    fprintf('==> Running RUSBoost (training period: %d-%d, testing period: %d-%d)...\n',2010,year_test-2,year_test-1,year_test);
    data_train = data_reader(file_input,'data_default',2010,2014);
    y_train = data_train.labels;
    X_train = data_train.features;
    paaer_train = data_train.paaers;

    % read testing data
    data_test = data_reader(file_input,'data_default',2015,2016);
    y_test = data_test.labels;
    X_test = data_test.features;
    paaer_test = unique(data_test.paaers(data_test.labels~=0));

    % handle serial frauds using PAAER
    if nopaar == 1
        y_train(ismember(paaer_train,paaer_test)) = 0;
    end

    % train model
    t1 = tic;
    t = templateTree('MinLeafSize',6); % base model
    rusboost = fitensemble(X_train,y_train,'RUSBoost',iters,t,'LearnRate',lrate,'RatioToSmallest',[small large]);
    t_train = toc(t1);

    % test model
    t2 = tic;
    [label_predict,dec_values] = predict(rusboost,X_test);
    dec_values = dec_values(:,2);
    t_test = toc(t2);

    % print performance results
    fprintf('Training time: %g seconds | Testing time %g seconds \n', t_train, t_test);
    for topN = [0.01, 0.02, 0.03, 0.04, 0.05]
        metrics = evaluate(y_test,label_predict,dec_values,topN);
        fprintf('Performance (top%d%% as cut-off thresh): \n',topN*100);
        fprintf('AUC: %.4f \n', metrics.auc);
        fprintf('F-score@k: %.4f \n', metrics.fscore_topk);
        fprintf('NCDG@k: %.4f \n', metrics.ndcg_at_k);
        fprintf('Sensitivity: %.2f%% \n', metrics.sensitivity_topk*100);
        fprintf('Precision: %.2f%% \n', metrics.precision_topk*100);

        new_row = {year_test, topN, metrics.auc, metrics.ndcg_at_k, metrics.sensitivity_topk, metrics.precision_topk,metrics.fscore_topk};
        results = [results; new_row];
    end

    diary off;

    % Write results to a CSV file
    output_file = strcat(file_path,file_output);
    writetable(results,output_file);

end

function result = print_test_rusboost_2022(file_input,file_output,file_path)
    iters = 300;
    lrate = 0.1;
    small = 1;
    large = 1;

    % financial 
    auc_values = [];
    fscore_values = [];
    ncdg_values = [];
    sensitivity_values = [];
    precision_values = [];
    Top_N = [];
    Test_year = [];

    % Initialize arrays to store results
    results = table(Test_year, Top_N, auc_values, ncdg_values, sensitivity_values, precision_values,fscore_values);



    diary("results_rusboost.txt");
    for year_test = 2017
        rng(0,'twister'); % fix random seed for reproducing the results
        % read training data
        fprintf('==> Running RUSBoost (training period: %d-%d, testing period: %d)...\n',2010,year_test-1,year_test);
        data_train = data_reader(file_input,'data_default',2010,year_test-1);
        y_train = data_train.labels;
        X_train = data_train.features;
        paaer_train = data_train.paaers;

        % read testing data
        data_test = data_reader(file_input,'data_default',year_test,year_test);
        y_test = data_test.labels;
        X_test = data_test.features;
        paaer_test = unique(data_test.paaers(data_test.labels~=0));

        % handle serial frauds using PAAER
        y_train(ismember(paaer_train,paaer_test)) = 0;

        % train model
        t1 = tic;
        t = templateTree('MinLeafSize',6); % base model
        rusboost = fitensemble(X_train,y_train,'RUSBoost',iters,t,'LearnRate',lrate,'RatioToSmallest',[small large]);
        t_train = toc(t1);

        % test model
        t2 = tic;
        [label_predict,dec_values] = predict(rusboost,X_test);
        dec_values = dec_values(:,2);
        t_test = toc(t2);


        % print performance results
        fprintf('Training time: %g seconds | Testing time %g seconds \n', t_train, t_test);
        for topN = [0.01, 0.02, 0.03, 0.04, 0.05]
            metrics = evaluate(y_test,label_predict,dec_values,topN);
            fprintf('Performance (top%d%% as cut-off thresh): \n',topN*100);
            fprintf('AUC: %.4f \n', metrics.auc);
            fprintf('F-score: %.4f \n', metrics.fscore_topk);
            fprintf('NCDG@k: %.4f \n', metrics.ndcg_at_k);
            fprintf('Sensitivity: %.2f%% \n', metrics.sensitivity_topk*100);
            fprintf('Precision: %.2f%% \n', metrics.precision_topk*100);
            new_row = {year_test, topN, metrics.auc,  metrics.ndcg_at_k, metrics.sensitivity_topk, metrics.precision_topk,metrics.fscore_topk};
            results = [results; new_row];
        end
    end
    diary off;
    output_file = strcat(file_path,file_output);
    % Write results to a CSV file
    writetable(results,output_file);

end