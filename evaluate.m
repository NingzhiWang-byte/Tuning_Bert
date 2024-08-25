%%%%%%%%%%%%%%%%%%%%%%%
% evaluation function %
%%%%%%%%%%%%%%%%%%%%%%%

function [results] = evaluate(label_true,label_predict,dec_values,topN)
pos_class = 1; % 1 as fraud label
neg_class = 0; % 0 as non-fraud label
assert(length(label_true)==length(label_predict));
assert(length(label_true)==length(dec_values));

% calculate metric: AUC
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(label_true,dec_values,pos_class,'negClass',neg_class);
results.auc = AUC;
results.auc_optimalPT = OPTROCPT;
results.roc_X = X;
results.roc_Y = Y;
plot(X,Y);
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC curve');

% calculate metric: sensitivity, specificity, and BAC (balanced accuracy) by using default cut-off thresh of classifier
tp=sum(label_true==pos_class & label_predict==pos_class);
fn=sum(label_true==pos_class & label_predict==neg_class);
tn=sum(label_true==neg_class & label_predict==neg_class);
fp=sum(label_true==neg_class & label_predict==pos_class);
sensitivity = tp/(tp+fn);
specificity = tn/(tn+fp);
results.bac = (sensitivity+specificity)/2;
results.sensitivity=sensitivity;
results.specificity=specificity;

% calculate metric: precision, sensitivity, specificity, and BAC (balanced accuracy) by using topN% cut-off thresh
k = round(length(label_true)*topN);
[~,idx] = sort(dec_values,'descend');
label_predict_topk = ones(length(label_true),1)*neg_class;
label_predict_topk(idx(1:k))=1;
tp_topk=sum(label_true==pos_class & label_predict_topk==pos_class);
fn_topk=sum(label_true==pos_class & label_predict_topk==neg_class);
tn_topk=sum(label_true==neg_class & label_predict_topk==neg_class);
fp_topk=sum(label_true==neg_class & label_predict_topk==pos_class);
sensitivity_topk = tp_topk/(tp_topk+fn_topk);
specificity_topk = tn_topk/(tn_topk+fp_topk);
results.bac_topk = (sensitivity_topk+specificity_topk)/2;
precision_topk = tp_topk/(tp_topk+fp_topk);
results.sensitivity_topk=sensitivity_topk;
results.specificity_topk=specificity_topk;
results.precision_topk = precision_topk;

% Calculate the number of hits and minimum of k and hits
hits = sum(label_true == pos_class);
kz = min(k, hits);

% calculate metric: F-score
fscore_topk = 2*precision_topk*sensitivity_topk/(precision_topk+sensitivity_topk);
results.fscore_topk = fscore_topk;

% calculate metric: MAP
label_true = label_true(:);
% initialize cumulative variables
cumPrecision = 0;
relevantCount = 0;

for i = 1:k
    if label_true(idx(i)) == 1
        relevantCount = relevantCount + 1;
        cumPrecision = cumPrecision + relevantCount / i;
    end
end

if relevantCount > 0
    AP = cumPrecision / relevantCount;
else
    AP = 0; % 如果没有相关文档，则AP为0
end

% 在这个例子中，我们只计算了一个查询的AP。
% 如果有多个查询，应该计算每个查询的AP，然后取平均值。
MAP = AP; % 这里只处理了单个查询的情况


results.map_topk = MAP;

% calculate metric: NDCG@k
% Assuming label_true, pos_class, k, and idx are already defined




% Calculate Ideal DCG (z)
z = 0.0;
for i = 1:kz
    rel = 1; % Assuming binary relevance
    z = z + (2^rel - 1) / log2(1 + i);
end

% Calculate DCG at k
dcg_at_k = 0.0;
for i = 1:k
    if label_true(idx(i)) == pos_class % Assuming pos_class indicates relevance
        rel = 1; % Assuming binary relevance
        dcg_at_k = dcg_at_k + (2^rel - 1) / log2(1 + i);
    end
end
% Calculate NDCG at k
if z ~= 0
    ndcg_at_k = dcg_at_k / z;
else
    ndcg_at_k = 0;
end

% Store result
results.ndcg_at_k = ndcg_at_k;

end
