%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is to replicate the results of RUSBoost model               %
% Our results are generated using Matlab R2020b on Windows 10           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Table 3 Table 5 %%%%%
file_input = 'env_winsored.csv';
file_output = '_LSTM_test_env.csv';
print_test_results_lstm(file_input,file_output);
file_input = 'noenv_winsored.csv';
file_output = '_LSTM_test_noenv.csv';
print_test_results_lstm(file_input,file_output);
file_input = 'csmar_winsored.csv';
file_output = '_LSTM_test_csmar.csv';
print_test_results_lstm(file_input,file_output);
file_input = 'financial_winsored.csv';
file_output = '_LSTM_test_financial.csv';
print_test_results_lstm(file_input,file_output);

%%%%% Table 4 %%%%%
file_input = 'envonly_winsored.csv';
file_output = 'LSTM_baseline_envonly.csv';
nopaar = 1;
print_LSTM_baseline(file_input,file_output,nopaar)
file_input = 'noenvonly_winsored.csv';
file_output = 'LSTM_baseline_noenvonly.csv';
nopaar = 1;
print_LSTM_baseline(file_input,file_output,nopaar)
file_input = 'csmaronly_winsored.csv';
file_output = 'LSTM_baseline_csmaronly.csv';
nopaar = 1;
print_LSTM_baseline(file_input,file_output,nopaar)

%%%%% Table 6 %%%%%
file_input = 'env_winsored.csv';
file_output = 'LSTM_baseline_env_nopaar.csv';
nopaar = 0;
print_LSTM_baseline(file_input,file_output,nopaar)
file_input = 'noenv_winsored.csv';
file_output = 'LSTM_baseline_noenv_nopaar.csv';
nopaar = 0;
print_LSTM_baseline(file_input,file_output,nopaar)
file_input = 'csmar_winsored.csv';
file_output = 'LSTM_baseline_csmar_nopaar.csv';
nopaar = 0;
print_LSTM_baseline(file_input,file_output,nopaar)
file_input = 'financial_winsored.csv';
file_output = 'LSTM_baseline_financial_nopaar.csv';
nopaar = 0;
print_LSTM_baseline(file_input,file_output,nopaar)


function result = print_test_results_lstm(file_input,file_output)
    warning off             % 关闭报警信息
    close all               % 关闭开启的图窗
    clc                     % 清空命令行

    % 循环每一个测试年份
    for year_test = 2016:2016
        clc;
        P_train = []; P_test = [];
        T_train = []; T_test = [];
        auc_values = [];
        ncdg_values = [];
        sensitivity_values = [];
        precision_values = [];
        Top_N = [];
        Test_year = [];

        % 初始化随机数生成器以确保结果可重复
        rng(0,'twister');
        % Initialize arrays to store results
        results = table(Test_year, Top_N, auc_values, ncdg_values, sensitivity_values, precision_values);

        fprintf('==> Running LSTM (training period: %d-%d, testing period:%d-%d)...\n', 2011, year_test-2, year_test-1,year_test);
        % 读取训练数据
        data_train = data_reader(file_input, 'data_default', 2011, year_test-2);
        P_train = data_train.features; % 特征数据
        T_train = data_train.labels; % 标签数据
        paaer_train = data_train.paaers;

        % 计算合适大小
        num_class = length(unique(data_train.labels));  % 类别数
        num_dim = size(data_train.features,2);      % 特征维度
        num_res = size(data_train.labels,1);            % 样本数

        % 读取测试数据
        data_test = data_reader(file_input, 'data_default', year_test-1, year_test);
        P_test = data_test.features; % 特征数据
        T_test = categorical(data_test.labels); % 标签数据
        paaer_test = unique(data_test.paaers(data_test.labels~=0));

        % handle serial greenwashing using PAAER
        T_train(ismember(paaer_train,paaer_test)) = 0;
        T_train = categorical(T_train);

        % 数据转置
        P_train = P_train'; P_test = P_test';
        T_train = T_train'; T_test = T_test';

        % 得到训练集和测试样本个数
        M = size(P_train, 2);
        N = size(P_test , 2);

        % 数据归一化
        [P_train, ps_input] = mapminmax(P_train, 0, 1);
        P_test  = mapminmax('apply', P_test, ps_input);
        t_train =  categorical(T_train)';
        t_test  =  categorical(T_test )';

        % 数据平铺
        P_train =  double(reshape(P_train, num_dim, 1, 1, M));
        P_test  =  double(reshape(P_test , num_dim, 1, 1, N));
        % 数据格式转化
        for i = 1 : M
            p_train{i, 1} = P_train(:, :, 1, i);
        end
        for i = 1 : N
            p_test{i, 1}  = P_test( :, :, 1, i);
        end



        % 建立模型
        layers = [
            sequenceInputLayer(num_dim)                                  % 输入层
            lstmLayer(6, 'OutputMode', 'last')                      % LSTM层
            reluLayer                                               % Relu激活层
            fullyConnectedLayer(num_class)                             % 全连接层（类别数） 
            softmaxLayer                                            % 分类层
            classificationLayer];

        % 参数设置
        options = trainingOptions('adam', ...       % Adam 梯度下降算法
            'MaxEpochs', 100, ...                  % 最大迭代次数
            'InitialLearnRate', 0.001, ...           % 初始学习率 原版：0.001
            'LearnRateSchedule', 'piecewise', ...   % 学习率下降
            'LearnRateDropFactor', 0.01, ...         % 学习率下降因子
            'LearnRateDropPeriod', 1000, ...         % 经过 750 次训练后 学习率为 0.01 * 0.1 原版：1000
            'Shuffle', 'every-epoch', ...           % 每次训练打乱数据集
            'ValidationPatience', Inf, ...          % 关闭验证
            'L2Regularization', 1e-4, ...           % 正则化参数
            'Plots', 'training-progress', ...       % 画出曲线
            'Verbose', false);

        % 训练 LSTM 模型
        net = trainNetwork(p_train, t_train, layers, options);
        t_sim1 = predict(net, p_train); 
        t_sim2 = predict(net, p_test );
        dec_values = t_sim2(:, 2);

        % get predicted labels
        T_sim1 = vec2ind(t_sim1')';
        label_predict = vec2ind(t_sim2')';
        T_test = double(T_test');

     for topN = [0.01, 0.02, 0.03, 0.04, 0.05]
            metrics = evaluate(data_test.labels,label_predict,dec_values,topN);
            fprintf('Performance (top%d%% as cut-off thresh): \n',topN*100);
            fprintf('AUC: %.4f \n', metrics.auc);
            fprintf('NCDG@k: %.4f \n', metrics.ndcg_at_k);
            fprintf('Sensitivity: %.2f%% \n', metrics.sensitivity_topk*100);
            fprintf('Precision: %.2f%% \n', metrics.precision_topk*100);
            new_row = {year_test, topN, metrics.auc, metrics.ndcg_at_k, metrics.sensitivity_topk, metrics.precision_topk};
            results = [results; new_row];
     end
     filename = sprintf('C:\\Users\\sz_wh\\Desktop\\【data】\\【毕业论文用】fraud论文复刻\\GreenwashingDetection\\results\\%d', year_test);
     output_file = strcat(filename,file_output);
     writetable(results, output_file, 'Delimiter', ',');

    end
end


function result = print_LSTM_baseline(file_input,file_output,nopaar)
    warning off             % 关闭报警信息
    close all               % 关闭开启的图窗

    P_train = []; P_test = [];
    T_train = []; T_test = [];

    auc_values = [];
    ncdg_values = [];
    sensitivity_values = [];
    precision_values = [];
    Top_N = [];
    Test_year = [];

    % Initialize arrays to store results
    results = table(Test_year, Top_N, auc_values, ncdg_values, sensitivity_values, precision_values);

    % 初始化随机数生成器以确保结果可重复
    rng(0,'twister');


    % 循环每一个测试年份
    year_test = 2015;
    fprintf('==> Running LSTM (training period: %d-%d, testing period: %d)...\n', 2011, year_test-1, year_test);
    % 读取训练数据
    data_train = data_reader(file_input, 'data_default', 2011, 2014);
    P_train = data_train.features; % 特征数据
    T_train = data_train.labels; % 标签数据
    paaer_train = data_train.paaers;

    % 计算合适大小
    num_class = length(unique(data_train.labels));  % 类别数
    num_dim = size(data_train.features,2);      % 特征维度
    num_res = size(data_train.labels,1);            % 样本数

    % 读取测试数据
    data_test = data_reader(file_input, 'data_default', 2015, 2016);
    P_test = data_test.features; % 特征数据
    T_test = categorical(data_test.labels); % 标签数据
    paaer_test = unique(data_test.paaers(data_test.labels~=0));

    % handle serial greenwashing using PAAER
    if nopaar == 1
        T_train(ismember(paaer_train,paaer_test)) = 0;
    end
    T_train = categorical(T_train);

    % 数据转置
    P_train = P_train'; P_test = P_test';
    T_train = T_train'; T_test = T_test';

    % 得到训练集和测试样本个数
    M = size(P_train, 2);
    N = size(P_test , 2);

    % 数据归一化
    [P_train, ps_input] = mapminmax(P_train, 0, 1);
    P_test  = mapminmax('apply', P_test, ps_input);
    t_train =  categorical(T_train)';
    t_test  =  categorical(T_test )';

    % 数据平铺
    P_train =  double(reshape(P_train, num_dim, 1, 1, M));
    P_test  =  double(reshape(P_test , num_dim, 1, 1, N));

    % 数据格式转化
    for i = 1 : M
        p_train{i, 1} = P_train(:, :, 1, i);
    end
    for i = 1 : N
        p_test{i, 1}  = P_test( :, :, 1, i);
    end



    % 建立模型
    layers = [
        sequenceInputLayer(num_dim)                                  % 输入层
        lstmLayer(6, 'OutputMode', 'last')                      % LSTM层
        reluLayer                                               % Relu激活层
        fullyConnectedLayer(num_class)                             % 全连接层（类别数） 
        softmaxLayer                                            % 分类层
        classificationLayer];


    % 参数设置
    options = trainingOptions('adam', ...       % Adam 梯度下降算法
        'MaxEpochs', 100, ...                  % 最大迭代次数
        'InitialLearnRate', 0.001, ...           % 初始学习率
        'LearnRateSchedule', 'piecewise', ...   % 学习率下降
        'LearnRateDropFactor', 0.01, ...         % 学习率下降因子
        'LearnRateDropPeriod', 1000, ...         % 经过 750 次训练后 学习率为 0.01 * 0.1
        'Shuffle', 'every-epoch', ...           % 每次训练打乱数据集
        'ValidationPatience', Inf, ...          % 关闭验证
        'L2Regularization', 1e-4, ...           % 正则化参数
        'Plots', 'training-progress', ...       % 画出曲线
        'Verbose', false);






    % 训练 LSTM 模型
    net = trainNetwork(p_train, t_train, layers, options);
    t_sim1 = predict(net, p_train); 
    t_sim2 = predict(net, p_test );
    dec_values = t_sim2(:, 2);


    % get predicted labels
    T_sim1 = vec2ind(t_sim1')';
    label_predict = vec2ind(t_sim2')';
    T_test = double(T_test');


    for topN = [0.01, 0.02, 0.03, 0.04, 0.05]
        metrics = evaluate(data_test.labels,label_predict,dec_values,topN);
        fprintf('Performance (top%d%% as cut-off thresh): \n',topN*100);
        fprintf('AUC: %.4f \n', metrics.auc);
        fprintf('NCDG@k: %.4f \n', metrics.ndcg_at_k);
        fprintf('Sensitivity: %.2f%% \n', metrics.sensitivity_topk*100);
        fprintf('Precision: %.2f%% \n', metrics.precision_topk*100);
        new_row = {year_test, topN, metrics.auc, metrics.ndcg_at_k, metrics.sensitivity_topk, metrics.precision_topk};
        results = [results; new_row];
    end
    file_path = 'C:\\Users\\sz_wh\\Desktop\\【data】\\【毕业论文用】fraud论文复刻\\GreenwashingDetection\\results\\';
    output_file = strcat(file_path,file_output);
    writetable(results, output_file, 'Delimiter', ',');
end


