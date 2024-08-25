%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is to replicate the results of SVM model                    %
% Our results are generated using Matlab R2020b on Windows 10           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%file_path = 'C:\\Users\\sz_wh\\Desktop\\【data】\\【毕业论文用-用行政处罚建立一个新的Y】\\data\\results\\';
file_path = 'C:\\Users\\sz_wh\\Desktop\\【data】\\【毕业论文用-用行政处罚建立一个新的Y】\\data\\results_southweekend\\';

%%%%%%%%%%%%% Logit %%%%%%%%%%%%%%%%%%%%
%%%%% Table 3 Table 5 %%%%%
file_input = 'env_winsored.csv';
file_output = 'logit_test_env3.csv';
print_Logit_test(file_input,file_output,file_path);
file_input = 'noenv_winsored.csv';
file_output = 'logit_test_noenv3.csv';
print_Logit_test(file_input,file_output,file_path);
file_input = 'financial_winsored.csv';
file_output = 'logit_test_financial3.csv';
print_Logit_test(file_input,file_output,file_path);

%%%% Table 6 %%%%
file_input = 'env_winsored.csv';
file_output = 'logit_baseline_env_nopaar.csv';
nopaar = 0;
print_baseline_result_logit(file_input,file_output,nopaar,file_path);
file_input = 'noenv_winsored.csv';
file_output = 'logit_baseline_noenv_nopaar.csv';
nopaar = 0;
print_baseline_result_logit(file_input,file_output,nopaar,file_path);
file_input = 'financial_winsored.csv';
file_output = 'logit_baseline_financial_nopaar.csv';
nopaar = 0;
print_baseline_result_logit(file_input,file_output,nopaar,file_path);
%%%%% 2022 %%%%%
file_input = 'env_winsored.csv';
file_output = 'SVM_2022_env.csv';
print_SVM_test_2022(file_input,file_output,file_path)
file_input = 'noenv_winsored.csv';
file_output = 'SVM_2022_noenv.csv';
print_SVM_test_2022(file_input,file_output,file_path)
file_input = 'financial_winsored.csv';
file_output = 'SVM_2022_financial.csv';
print_SVM_test_2022(file_input,file_output,file_path)

file_input = 'env_winsored.csv';
file_output = 'Logit_2022_env.csv';
print_Logit_test_2022(file_input,file_output,file_path)
file_input = 'noenv_winsored.csv';
file_output = 'Logit_2022_noenv.csv';
print_Logit_test_2022(file_input,file_output,file_path)
file_input = 'financial_winsored.csv';
file_output = 'Logit_2022_financial.csv';
print_Logit_test_2022(file_input,file_output,file_path)

%%%%%%%%%%%%%%%%%%%%%%%%% SVM %%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Table 3 Table 5 %%%%%
file_input = 'env_winsored.csv';
file_output = 'SVM_test_env3.csv';
print_SVM_test(file_input,file_output,file_path)

file_input = 'noenv_winsored.csv';
file_output = 'SVM_test_noenv3.csv';
print_SVM_test(file_input,file_output,file_path)

file_input = 'financial_winsored.csv';
file_output = 'SVM_test_financial3.csv';
print_SVM_test(file_input,file_output,file_path)


%%%% Table 6 %%%%
file_input = 'env_winsored.csv';
file_output = 'SVM_baseline_env_nopaar.csv';
nopaar = 0;
print_SVM_Baseline(file_input,file_output,nopaar,file_path);

file_input = 'noenv_winsored.csv';
file_output = 'SVM_baseline_noenv_nopaar.csv';
nopaar = 0;
print_SVM_Baseline(file_input,file_output,nopaar,file_path);

file_input = 'financial_winsored.csv';
file_output = 'SVM_baseline_financial_nopaar.csv';
nopaar = 0;
print_SVM_Baseline(file_input,file_output,nopaar,file_path);




function result = print_SVM_test(file_input,file_output,file_path)
    auc_values = [];
    ncdg_values = [];
    sensitivity_values = [];
    precision_values = [];
    fscore_values = [];
    Top_N = [];
    Test_year = [];

    % Initialize arrays to store results
    results = table(Test_year, Top_N, auc_values, ncdg_values, sensitivity_values, precision_values,fscore_values);

    diary("results_svm.txt");
    for year_test = 2014:2016
        rng(0,'twister'); % fix random seed for reproducing the results
        % read training data
        fprintf('==> Running SVM model (training period: %d-%d, testing period: %d)...\n',2010,year_test-2,year_test);
        data_train = data_reader(file_input,'data_default',2010,year_test-2);
        y_train = data_train.labels;
        X_train = data_train.features;
        paaer_train = data_train.paaers;

        % read testing data
        data_test = data_reader(file_input,'data_default',year_test-1,year_test);
        y_test = data_test.labels;
        X_test = data_test.features;
        paaer_test = unique(data_test.paaers(data_test.labels~=0));

        % handle serial greenwashing using PAAER
        y_train(ismember(paaer_train,paaer_test)) = 0;

        % train model
        t1 = tic;
        svm_model = fitcsvm(X_train, y_train, 'Standardize', true, 'KernelFunction', 'RBF');
        %svm_model = fitcsvm(X_train, y_train, 'Standardize', true, 'KernelFunction', 'linear', 'ClassNames', [0, 1]);
        t_train = toc(t1);

        % test model
        t2 = tic;
        label_predict = predict(svm_model, X_test);
        [~, scores] = predict(svm_model, X_test);
        t_test = toc(t2);

        % print performance results
        fprintf('Training time: %g seconds | Testing time %g seconds \n', t_train, t_test);
        for topN = [0.01, 0.02, 0.03, 0.04, 0.05]
            metrics = evaluate(y_test,label_predict,scores(:, 2),topN);
            fprintf('Performance (top%d%% as cut-off thresh): \n',topN*100);
            fprintf('AUC: %.4f \n', metrics.auc);
            fprintf('NCDG@k: %.4f \n', metrics.ndcg_at_k);
            fprintf('Sensitivity: %.2f%% \n', metrics.sensitivity_topk*100);
            fprintf('Precision: %.2f%% \n', metrics.precision_topk*100);

            new_row = {year_test, topN, metrics.auc, metrics.ndcg_at_k, metrics.sensitivity_topk, metrics.precision_topk,metrics.fscore_topk};
            results = [results; new_row];
        end
    end
    % Write results to a CSV file
    output_file = strcat(file_path,file_output);
    writetable(results,output_file);
    
end


function result = print_SVM_Baseline(file_input, file_output,nopaar,file_path)
    auc_values = [];
    ncdg_values = [];
    sensitivity_values = [];
    precision_values = [];
    fscore_values = [];
    Top_N = [];
    Test_year = [];

    % Initialize arrays to store results
    results = table(Test_year, Top_N, auc_values, ncdg_values, sensitivity_values, precision_values,fscore_values);

    diary("results_svm.txt");
    year_test = 2016;
    rng(0,'twister'); % fix random seed for reproducing the results
    % read training data
    fprintf('==> Running SVM model (training period: %d-%d, testing period: %d, with %d-year gap)...\n',2010,year_test-2,year_test,2);
    data_train = data_reader(file_input,'data_default',2010,year_test-2);
    y_train = data_train.labels;
    X_train = data_train.features;
    paaer_train = data_train.paaers;

    % read testing data
    data_test = data_reader(file_input,'data_default',year_test-1,year_test);
    y_test = data_test.labels;
    X_test = data_test.features;
    paaer_test = unique(data_test.paaers(data_test.labels~=0));

    % handle serial greenwashing using PAAER
    if nopaar == 1
        y_train(ismember(paaer_train,paaer_test)) = 0;
    end
    
    % train model
    t1 = tic;
    svm_model = fitcsvm(X_train, y_train, 'Standardize', true, 'KernelFunction', 'RBF');
    %svm_model = fitcsvm(X_train, y_train, 'Standardize', true, 'KernelFunction', 'linear', 'ClassNames', [0, 1]);
    t_train = toc(t1);

    % test model
    t2 = tic;
    label_predict = predict(svm_model, X_test);
    [~, scores] = predict(svm_model, X_test);
    t_test = toc(t2);

    % print performance results
    fprintf('Training time: %g seconds | Testing time %g seconds \n', t_train, t_test);
    for topN = [0.01, 0.02, 0.03, 0.04, 0.05]
        metrics = evaluate(y_test,label_predict,scores(:, 2),topN);
        fprintf('Performance (top%d%% as cut-off thresh): \n',topN*100);
        fprintf('AUC: %.4f \n', metrics.auc);
        fprintf('NCDG@k: %.4f \n', metrics.ndcg_at_k);
        fprintf('Sensitivity: %.2f%% \n', metrics.sensitivity_topk*100);
        fprintf('Precision: %.2f%% \n', metrics.precision_topk*100);
        new_row = {year_test, topN, metrics.auc, metrics.ndcg_at_k, metrics.sensitivity_topk, metrics.precision_topk, metrics.fscore_topk};
        results = [results; new_row];
    end

    % Write results to a CSV file
    output_file = strcat(file_path,file_output);
    writetable(results,output_file);
end



function result = print_SVM_test_2022(file_input,file_output,file_path)
    auc_values = [];
    ncdg_values = [];
    sensitivity_values = [];
    precision_values = [];
    fscore_values = [];
    Top_N = [];
    Test_year = [];

    % Initialize arrays to store results
    results = table(Test_year, Top_N, auc_values, ncdg_values, sensitivity_values, precision_values,fscore_values);

    diary("results_svm.txt");
    for year_test = 2017
        rng(0,'twister'); % fix random seed for reproducing the results
        % read training data
        fprintf('==> Running SVM model (training period: %d-%d, testing period: %d)...\n',2010,year_test-1,year_test);
        data_train = data_reader(file_input,'data_default',2010,year_test-1);
        y_train = data_train.labels;
        X_train = data_train.features;
        paaer_train = data_train.paaers;

        % read testing data
        data_test = data_reader(file_input,'data_default',year_test,year_test);
        y_test = data_test.labels;
        X_test = data_test.features;
        paaer_test = unique(data_test.paaers(data_test.labels~=0));

        % handle serial greenwashing using PAAER
        y_train(ismember(paaer_train,paaer_test)) = 0;

        % train model
        t1 = tic;
        svm_model = fitcsvm(X_train, y_train, 'Standardize', true, 'KernelFunction', 'RBF');
        %svm_model = fitcsvm(X_train, y_train, 'Standardize', true, 'KernelFunction', 'linear', 'ClassNames', [0, 1]);
        t_train = toc(t1);

        % test model
        t2 = tic;
        label_predict = predict(svm_model, X_test);
        [~, scores] = predict(svm_model, X_test);
        t_test = toc(t2);

        % print performance results
        fprintf('Training time: %g seconds | Testing time %g seconds \n', t_train, t_test);
        for topN = [0.01, 0.02, 0.03, 0.04, 0.05]
            metrics = evaluate(y_test,label_predict,scores(:, 2),topN);
            fprintf('Performance (top%d%% as cut-off thresh): \n',topN*100);
            fprintf('AUC: %.4f \n', metrics.auc);
            fprintf('NCDG@k: %.4f \n', metrics.ndcg_at_k);
            fprintf('Sensitivity: %.2f%% \n', metrics.sensitivity_topk*100);
            fprintf('Precision: %.2f%% \n', metrics.precision_topk*100);

            new_row = {year_test, topN, metrics.auc, metrics.ndcg_at_k, metrics.sensitivity_topk, metrics.precision_topk,metrics.fscore_topk};
            results = [results; new_row];
        end
    end
    % Write results to a CSV file
    output_file = strcat(file_path,file_output);
    writetable(results,output_file);
    
end



function result = print_Logit_test(file_input,file_output,file_path)
    auc_values = [];
    ncdg_values = [];
    sensitivity_values = [];
    precision_values = [];
    fscore_values = [];
    Top_N = [];
    Test_year = [];

    % Initialize arrays to store results
    results = table(Test_year, Top_N, auc_values, ncdg_values, sensitivity_values, precision_values,fscore_values);

    diary("results_logit.txt");
    for year_test = 2014:2016
        rng(0,'twister'); % fix random seed for reproducing the results
        % read training data
        fprintf('==> Running Logit model (training period: %d-%d, testing period: %d-%d)...\n',2010,year_test-2,year_test-1,year_test);
        data_train = data_reader(file_input,'data_default',2010,year_test-2);
        y_train = data_train.labels;
        X_train = data_train.features;
        paaer_train = data_train.paaers;

        % read testing data
        data_test = data_reader(file_input,'data_default',year_test-1,year_test);
        y_test = data_test.labels;
        X_test = data_test.features;
        paaer_test = unique(data_test.paaers(data_test.labels~=0));

        % handle serial greenwashing using PAAER
        y_train(ismember(paaer_train,paaer_test)) = 0;

        % train model
        t1 = tic;
        %logit_model = fitglm(X_train,y_train,'Distribution','binomial');
        % 计算每个类别的样本数量
        num_total = length(y_train);
        num_positive = sum(y_train == 1);
        num_negative = num_total - num_positive;

        % 计算权重
        weight_positive = num_total / (2 * num_positive); % 少数类权重
        weight_negative = num_total / (2 * num_negative); % 多数类权重

        % 创建权重向量
        weights_vector = ones(num_total, 1);
        weights_vector(y_train == 1) = weight_positive;   % 为正样本分配权重
        weights_vector(y_train == 0) = weight_negative;   % 为负样本分配权重

        logit_model = fitglm(X_train,y_train,'Distribution','binomial', 'Link', 'logit');
        
        t_train = toc(t1);

        % test model
        t2 = tic;
        [label_predict, ~] = predict(logit_model,X_test);
        t_test = toc(t2);
        dec_values = predict(logit_model, X_test);

        % print performance results
        fprintf('Training time: %g seconds | Testing time %g seconds \n', t_train, t_test);
        for topN = [0.01, 0.02, 0.03, 0.04, 0.05]
            metrics = evaluate(y_test,label_predict,dec_values,topN);
            fprintf('Performance (top%d%% as cut-off thresh): \n',topN*100);
            fprintf('AUC: %.4f \n', metrics.auc);
            fprintf('NCDG@k: %.4f \n', metrics.ndcg_at_k);
            fprintf('Sensitivity: %.2f%% \n', metrics.sensitivity_topk*100);
            fprintf('Precision: %.2f%% \n', metrics.precision_topk*100);

            new_row = {year_test, topN, metrics.auc, metrics.ndcg_at_k, metrics.sensitivity_topk, metrics.precision_topk,metrics.fscore_topk};
            results = [results; new_row];
        end
    end
    % Write results to a CSV file
    output_file = strcat(file_path,file_output);
    writetable(results,output_file);
end

function result = print_baseline_result_logit(file_input,file_output,nopaar,file_path)
    auc_values = [];
    ncdg_values = [];
    sensitivity_values = [];
    precision_values = [];
    fscore_values = [];
    Top_N = [];
    Test_year = [];

    % Initialize arrays to store results
    results = table(Test_year, Top_N, auc_values, ncdg_values, sensitivity_values, precision_values,fscore_values);

    diary("results_logit.txt");
    year_test = 2016;
    rng(0,'twister'); % fix random seed for reproducing the results
    % read training data
    fprintf('==> Running Logit model (training period: %d-%d, testing period: %d)...\n',2010,year_test-1,year_test);
    data_train = data_reader(file_input,'data_default',2010,year_test-2);
    y_train = data_train.labels;
    X_train = data_train.features;
    paaer_train = data_train.paaers;

    % read testing data
    data_test = data_reader(file_input,'data_default',year_test-1,year_test);
    y_test = data_test.labels;
    X_test = data_test.features;
    paaer_test = unique(data_test.paaers(data_test.labels~=0));

    % handle serial greenwashing using PAAER
    if nopaar == 1
        y_train(ismember(paaer_train,paaer_test)) = 0;
    end
    % train model
    t1 = tic;
    
    % 计算每个类别的样本数量
    num_total = length(y_train);
    num_positive = sum(y_train == 1);
    num_negative = num_total - num_positive;

    % 计算权重
    weight_positive = num_total / (2 * num_positive); % 少数类权重
    weight_negative = num_total / (2 * num_negative); % 多数类权重

    % 创建权重向量
    weights_vector = ones(num_total, 1);
    weights_vector(y_train == 1) = weight_positive;   % 为正样本分配权重
    weights_vector(y_train == 0) = weight_negative;   % 为负样本分配权重

    logit_model = fitglm(X_train,y_train,'Distribution','binomial', 'Link', 'logit');

    t_train = toc(t1);

    % test model
    t2 = tic;
    [label_predict, ~] = predict(logit_model,X_test);
    t_test = toc(t2);
    dec_values = predict(logit_model, X_test);

    % print performance results
    fprintf('Training time: %g seconds | Testing time %g seconds \n', t_train, t_test);
    for topN = [0.01, 0.02, 0.03, 0.04, 0.05]
        metrics = evaluate(y_test,label_predict,dec_values,topN);
        fprintf('Performance (top%d%% as cut-off thresh): \n',topN*100);
        fprintf('AUC: %.4f \n', metrics.auc);
        fprintf('NCDG@k: %.4f \n', metrics.ndcg_at_k);
        fprintf('Sensitivity: %.2f%% \n', metrics.sensitivity_topk*100);
        fprintf('Precision: %.2f%% \n', metrics.precision_topk*100);

        new_row = {year_test, topN, metrics.auc, metrics.ndcg_at_k, metrics.sensitivity_topk, metrics.precision_topk,metrics.fscore_topk};
        results = [results; new_row];
    end

    % Write results to a CSV file
    output_file = strcat(file_path,file_output);
    writetable(results,output_file);

end

function result = print_Logit_test_2022(file_input,file_output,file_path)
    auc_values = [];
    ncdg_values = [];
    sensitivity_values = [];
    precision_values = [];
    fscore_values = [];
    Top_N = [];
    Test_year = [];

    % Initialize arrays to store results
    results = table(Test_year, Top_N, auc_values, ncdg_values, sensitivity_values, precision_values,fscore_values);

    diary("results_logit.txt");
    for year_test = 2017
        rng(0,'twister'); % fix random seed for reproducing the results
        % read training data
        fprintf('==> Running Logit model (training period: %d-%d, testing period: %d-%d)...\n',2010,year_test-1,year_test,year_test);
        data_train = data_reader(file_input,'data_default',2010,year_test-1);
        y_train = data_train.labels;
        X_train = data_train.features;
        paaer_train = data_train.paaers;

        % read testing data
        data_test = data_reader(file_input,'data_default',year_test,year_test);
        y_test = data_test.labels;
        X_test = data_test.features;
        paaer_test = unique(data_test.paaers(data_test.labels~=0));

        % handle serial greenwashing using PAAER
        y_train(ismember(paaer_train,paaer_test)) = 0;

        % train model
        t1 = tic;
        
        % 计算每个类别的样本数量
        num_total = length(y_train);
        num_positive = sum(y_train == 1);
        num_negative = num_total - num_positive;

        % 计算权重
        weight_positive = num_total / (2 * num_positive); % 少数类权重
        weight_negative = num_total / (2 * num_negative); % 多数类权重

        % 创建权重向量
        weights_vector = ones(num_total, 1);
        weights_vector(y_train == 1) = weight_positive;   % 为正样本分配权重
        weights_vector(y_train == 0) = weight_negative;   % 为负样本分配权重

        %logit_model = fitglm(X_train,y_train,'Distribution','binomial');
        logit_model = fitglm(X_train,y_train,'Distribution','binomial', 'Link', 'logit');

        t_train = toc(t1);

        % test model
        t2 = tic;
        [label_predict, ~] = predict(logit_model,X_test);
        t_test = toc(t2);
        dec_values = predict(logit_model, X_test);

        % print performance results
        fprintf('Training time: %g seconds | Testing time %g seconds \n', t_train, t_test);
        for topN = [0.01, 0.02, 0.03, 0.04, 0.05]
            metrics = evaluate(y_test,label_predict,dec_values,topN);
            fprintf('Performance (top%d%% as cut-off thresh): \n',topN*100);
            fprintf('AUC: %.4f \n', metrics.auc);
            fprintf('NCDG@k: %.4f \n', metrics.ndcg_at_k);
            fprintf('Sensitivity: %.2f%% \n', metrics.sensitivity_topk*100);
            fprintf('Precision: %.2f%% \n', metrics.precision_topk*100);

            new_row = {year_test, topN, metrics.auc, metrics.ndcg_at_k, metrics.sensitivity_topk, metrics.precision_topk,metrics.fscore_topk};
            results = [results; new_row];
        end
    end
    % Write results to a CSV file
    output_file = strcat(file_path,file_output);
    writetable(results,output_file);
end