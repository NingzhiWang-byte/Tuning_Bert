%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dater_reader: the function for reading fraud data in csv format %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [data] = data_reader(data_path, data_format, year_start, year_end)
data_table = readtable(data_path);
switch data_format
    case 'data_default' % read data with 23 raw accounting variables
        data.years = data_table.fyear;
        idx = data.years>=year_start & data.years<=year_end;
        data.years = data.years(idx);
        data.firms = data_table.gvkey(idx);
        data.paaers = data_table.p_aaers(idx);
        data.labels = data_table.misstate(idx);
        data.features = table2array(data_table(idx, 5:end));
        data.num_obervations = size(data.features,1);
        data.num_features = size(data.features,2);
        % extract variable names
        var_names = data_table.Properties.VariableNames;
        data.var_names = var_names(5:end); % assuming the variable names start from column 5
    otherwise
        disp('Error: unsupported data format!');
end

fprintf('Data Loaded: %s, %d features, %d observations.\n',data_path, data.num_features, data.num_obervations);

end


%{
function [data] = data_reader(data_path, data_format, year_start, year_end)
temp = csvread(data_path, 1, 0);
switch data_format
    case 'data_default' % read data with 55 raw accounting variables
        data.years = temp(:, 1);
        idx = data.years>=year_start & data.years<=year_end;
        data.years = temp(idx, 1);
        data.firms = temp(idx, 2);
        data.paaers = temp(idx, 3);
        data.labels = temp(idx, 4);
        data.features = temp(idx,5:60);
        data.num_obervations = size(data.features,1);
        data.num_features = size(data.features,2);
        % add variable names
        var_names = {'var1', 'var2', 'var3', ...};
        % variable names
        data.var_names = var_names(5:60);
    otherwise
        disp('Error: unsupported data format!');
end

fprintf('Data Loaded: %s, %d features, %d observations.\n',data_path, data.num_features, data.num_obervations);

end
%}
