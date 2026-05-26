%% 预测性维护课程报告实验脚本（MATLAB）
% 本脚本用于完成 AI4I 2020 预测性维护数据集上的分类实验，
% 主要工作包括：
% 1. 读取并预处理数据；
% 2. 训练四种分类模型；
% 3. 进行交叉验证和保持集评估；
% 4. 计算多种机器学习评价指标；
% 5. 输出表格和可视化图像

clear;
clc;

if exist('maxNumCompThreads', 'file') == 2
    maxNumCompThreads(1);
end

% 固定随机种子，保证每次运行结果尽量可复现。
rng(2026, 'twister');

%% 一、实验配置
% 使用结构体集中管理所有路径和实验参数，便于后续修改。
config = struct();
config.projectRoot = pwd;
config.dataFile = fullfile(config.projectRoot, 'datasets', 'predictive_maintenance', ...
    'ai4i', 'ai4i2020.csv');
config.outputDir = fullfile(config.projectRoot, 'outputs');
config.showFigures = true;

config.mode = "quick";  

% 根据模式设置不同的交叉验证折数和模型复杂度。
switch config.mode
    case "quick"
        config.numFolds = 10;
        config.treeMaxSplits = 20;
        config.treeMinLeafSize = 10;
        config.bagCycles = 40;
        config.rusCycles = 60;
    case "full"
        config.numFolds = 10;
        config.treeMaxSplits = 40;
        config.treeMinLeafSize = 5;
        config.bagCycles = 80;
        config.rusCycles = 120;
    otherwise
        error('Unknown config.mode: %s', config.mode);
end

% 如果输出目录不存在，则自动创建。
if ~isfolder(config.outputDir)
    mkdir(config.outputDir);
end

fprintf('Running in %s mode.\n', config.mode);
fprintf('Data file: %s\n', config.dataFile);

%% 二、读取数据并完成预处理
% 读取原始表格数据，同时保留原始列名格式。
rawTbl = readtable(config.dataFile, 'VariableNamingRule', 'preserve');

% prepareData 函数负责：
% 1. 删除会造成标签泄漏的字段；
% 2. 提取特征和标签；
% 3. 对数值特征标准化；
% 4. 对类别特征做哑变量编码。
[featureTbl, featureMatrix, target, featureNames] = prepareData(rawTbl);

%% 三、交叉验证主实验
% 使用 K 折交叉验证评估四种模型的平均性能。
cvp = cvpartition(target, 'KFold', config.numFolds);
foldResults = table();

for foldIdx = 1:config.numFolds
    fprintf('\n===== Fold %d / %d =====\n', foldIdx, config.numFolds);

    % 获取当前折的训练集和测试集索引。
    trainIdx = training(cvp, foldIdx);
    testIdx = test(cvp, foldIdx);

    % 同时保留表格形式和数值矩阵形式的数据，
    % 因为不同模型的输入格式要求不同。
    XTrainTable = featureTbl(trainIdx, :);
    XTestTable = featureTbl(testIdx, :);
    XTrainMatrix = featureMatrix(trainIdx, :);
    XTestMatrix = featureMatrix(testIdx, :);
    yTrain = target(trainIdx);
    yTest = target(testIdx);

    % 计算正类样本权重，用于缓解类别不平衡问题。
    obsWeights = makePositiveClassWeights(yTrain);

    % 训练四种模型：Logistic、决策树、Bagging、RUSBoost。
    models = trainModels(XTrainTable, XTrainMatrix, yTrain, obsWeights, config);

    % 依次对每个模型进行预测并记录指标。
    for modelIdx = 1:numel(models)
        modelName = models(modelIdx).Name;
        modelObject = models(modelIdx).Model;
        inputKind = models(modelIdx).InputKind;

        % 根据模型输入类型，调用统一的预测函数。
        [predLabel, positiveScore] = predictWithScore(modelObject, inputKind, ...
            XTestTable, XTestMatrix);

        % 计算当前模型在当前折上的评价指标。
        metrics = computeBinaryMetrics(yTest, predLabel, positiveScore);

        % 将本折实验结果追加到总表中，便于后续汇总。
        newRow = table(string(modelName), foldIdx, metrics.Accuracy, metrics.Precision, ...
            metrics.Recall, metrics.Specificity, metrics.F1, metrics.F2, ...
            metrics.GMean, metrics.MCC, metrics.ROCAUC, metrics.PRAUC, ...
            metrics.AP, metrics.BalancedAccuracy, ...
            'VariableNames', {'Model', 'Fold', 'Accuracy', 'Precision', 'Recall', ...
            'Specificity', 'F1', 'F2', 'GMean', 'MCC', 'ROCAUC', 'PRAUC', ...
            'AP', 'BalancedAccuracy'});
        foldResults = [foldResults; newRow]; %#ok<AGROW>

        % 在命令行输出关键指标，便于快速查看结果。
        fprintf(['%-16s Acc %.4f | F1 %.4f | F2 %.4f | Recall %.4f | ' ...
            'ROC-AUC %.4f | PR-AUC %.4f | AP %.4f\n'], ...
            modelName, metrics.Accuracy, metrics.F1, metrics.F2, ...
            metrics.Recall, metrics.ROCAUC, metrics.PRAUC, metrics.AP);
    end
end

%% 四、汇总交叉验证结果
% 对每个模型在所有折上的结果求平均。
summaryResults = groupsummary(foldResults, 'Model', 'mean', ...
    {'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'F2', 'GMean', ...
    'MCC', 'ROCAUC', 'PRAUC', 'AP', 'BalancedAccuracy'});

% 这里按 mean_F1 从高到低排序，便于找出综合表现最好的模型。
summaryResults = sortrows(summaryResults, 'mean_F1', 'descend');

%% 五、保存表格结果
writetable(foldResults, fullfile(config.outputDir, 'fold_metrics.csv'));
writetable(summaryResults, fullfile(config.outputDir, 'summary_metrics.csv'));

%% 六、生成实验说明和图像
% 写出一份简要的文本说明，记录数据规模、特征和模型排序。
writeRunNotes(config, featureNames, rawTbl, summaryResults);

% 生成各类总览图。
plotMetricComparison(summaryResults, config.outputDir, config);
plotErrorCurve(foldResults, config.outputDir, config);
plotDatasetOverview(rawTbl, config.outputDir, config);

%% 七、保持集分析
% 除交叉验证外，再单独做一次保持集分析，
% 用于生成混淆矩阵、ROC/PR 曲线等更直观的图。
allModelAnalysis = analyzeAllModels(featureTbl, featureMatrix, target, featureNames, config);

% 保存所有模型在保持集上的评价结果。
saveAllModelHoldoutMetrics(allModelAnalysis, config.outputDir);

% 生成四种模型的混淆矩阵、ROC/PR 对比图和增强指标热力图。
plotAllModelConfusions(allModelAnalysis, config.outputDir, config);
plotRankingCurvesComparison(allModelAnalysis, config.outputDir, config);
plotAdvancedMetricHeatmap(summaryResults, config.outputDir, config);

%% 八、最佳模型的单独分析
% 默认选取交叉验证中 F1 平均值最高的模型作为最佳模型。
bestModelName = string(summaryResults.Model(1));
bestModelAnalysis = allModelAnalysis(bestModelName);

% 输出最佳模型说明，并生成其单独诊断图。
writeBestModelNotes(bestModelAnalysis, config.outputDir);
plotBestModelDiagnostics(bestModelAnalysis, config.outputDir, config);

%% 九、命令行输出最终结果
fprintf('\nDone. Results saved to:\n%s\n', config.outputDir);
disp(summaryResults(:, {'Model', 'mean_F1', 'mean_F2', 'mean_MCC', ...
    'mean_Recall', 'mean_ROCAUC', 'mean_PRAUC', 'mean_AP', ...
    'mean_BalancedAccuracy'}));

%% ======================== 本地函数区 ========================

function [featureTbl, featureMatrix, target, featureNames] = prepareData(rawTbl)
    % 该函数用于完成数据预处理：
    % 1. 删除会泄漏标签的信息；
    % 2. 分离特征和目标值；
    % 3. 标准化数值特征；
    % 4. 将类别特征转成哑变量。

    % 这些列会直接或间接暴露故障标签，因此不能用于训练。
    leakVars = {'UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'};
    keepVars = setdiff(rawTbl.Properties.VariableNames, leakVars, 'stable');
    tbl = rawTbl(:, keepVars);

    % 将产品类型转为类别变量。
    tbl.Type = categorical(tbl.Type);

    % 目标列是 Machine failure。
    target = categorical(tbl.("Machine failure"));

    % 去掉标签列后得到纯特征表。
    featureTbl = removevars(tbl, {'Machine failure'});

    % 挑出数值特征列，后续用于标准化和矩阵输入。
    numericVars = {'Air temperature [K]', 'Process temperature [K]', ...
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'};
    numericMatrix = featureTbl{:, numericVars};

    % 标准化数值特征，减少量纲不同带来的影响。
    numericMatrix = normalize(numericMatrix);

    % 将类别型的 Type 转成哑变量。
    typeDummy = dummyvar(featureTbl.Type);

    % 拼接数值特征和哑变量，形成模型可直接使用的数值矩阵。
    featureMatrix = [numericMatrix, typeDummy];

    % 构造特征名列表，便于后续做特征重要性分析。
    typeCategories = categories(featureTbl.Type);
    typeNames = "Type_" + string(typeCategories)';
    featureNames = [string(numericVars), typeNames];
end

function obsWeights = makePositiveClassWeights(yTrain)
    % 该函数根据训练集的类别分布，为少数类（故障类）分配更高权重。
    % 这样做的目的，是缓解正常样本远多于故障样本带来的偏置。

    yNumeric = double(yTrain) - 1;
    negativeCount = sum(yNumeric == 0);
    positiveCount = sum(yNumeric == 1);

    % 正类权重近似设为“负类样本数 / 正类样本数”。
    positiveWeight = max(1, negativeCount / max(positiveCount, 1));
    obsWeights = ones(size(yNumeric));
    obsWeights(yNumeric == 1) = positiveWeight;
end

function models = trainModels(XTrainTable, XTrainMatrix, yTrain, obsWeights, config)
    % 该函数统一训练四种模型，并将模型对象存入结构体数组返回。
    % 不同模型对输入格式的要求不同，因此同时保留表格和矩阵版本。

    % 为树类模型设置统一的基础模板，避免不同模型差异太大。
    treeTemplate = templateTree('MaxNumSplits', config.treeMaxSplits, ...
        'MinLeafSize', config.treeMinLeafSize);

    % 1) Logistic 回归：基础线性分类模型。
    models(1).Name = "Logistic";
    models(1).InputKind = "matrix";
    models(1).Model = fitclinear(XTrainMatrix, yTrain, ...
        'Learner', 'logistic', ...
        'ObservationsIn', 'rows', ...
        'ClassNames', categorical([0; 1]), ...
        'Weights', obsWeights);

    % 2) 决策树：结构清晰、可解释性强。
    models(2).Name = "DecisionTree";
    models(2).InputKind = "table";
    models(2).Model = fitctree(XTrainTable, yTrain, ...
        'MaxNumSplits', config.treeMaxSplits, ...
        'MinLeafSize', config.treeMinLeafSize, ...
        'Weights', obsWeights, ...
        'PredictorSelection', 'allsplits', ...
        'ClassNames', categorical([0; 1]));

    % 3) Bagging 集成树：通过多棵树降低方差，提高稳定性。
    models(3).Name = "BaggedTrees";
    models(3).InputKind = "table";
    models(3).Model = fitcensemble(XTrainTable, yTrain, ...
        'Method', 'Bag', ...
        'NumLearningCycles', config.bagCycles, ...
        'Learners', treeTemplate, ...
        'Weights', obsWeights, ...
        'ClassNames', categorical([0; 1]));

    % 4) RUSBoost：对不平衡分类更友好的提升方法。
    models(4).Name = "RUSBoost";
    models(4).InputKind = "table";
    models(4).Model = fitcensemble(XTrainTable, yTrain, ...
        'Method', 'RUSBoost', ...
        'NumLearningCycles', config.rusCycles, ...
        'Learners', treeTemplate, ...
        'ClassNames', categorical([0; 1]));
end

function [predLabel, positiveScore] = predictWithScore(modelObject, inputKind, ...
    XTestTable, XTestMatrix)
    % 统一预测接口：
    % predLabel 为预测类别；
    % positiveScore 为属于正类（故障类）的得分或概率。

    if inputKind == "table"
        [predLabel, score] = predict(modelObject, XTestTable);
    else
        [predLabel, score] = predict(modelObject, XTestMatrix);
    end

    predLabel = categorical(predLabel);

    % 找出 score 中对应正类（类别 1）的那一列。
    classNames = categorical(modelObject.ClassNames);
    positiveColumn = find(classNames == categorical(1), 1);
    if isempty(positiveColumn)
        error('Positive class column was not found in score output.');
    end
    positiveScore = score(:, positiveColumn);
end

function metrics = computeBinaryMetrics(yTrue, yPred, positiveScore)
    % 该函数同时计算：
    % 1. 基于混淆矩阵的指标；
    % 2. 基于排序能力的指标（ROC-AUC、PR-AUC、AP）。

    yTrue = double(categorical(yTrue)) - 1;
    yPred = double(categorical(yPred)) - 1;

    % 先计算由 TP/TN/FP/FN 决定的常用指标。
    metrics = computeConfusionMetrics(yTrue, yPred);

    % ROC-AUC 衡量整体区分正负类的能力。
    [~, ~, ~, metrics.ROCAUC] = perfcurve(yTrue, positiveScore, 1);

    % PR 曲线更适合不平衡分类任务。
    [recallCurve, precisionCurve] = perfcurve(yTrue, positiveScore, 1, ...
        'xCrit', 'reca', 'yCrit', 'prec');

    % 去掉 NaN 点，避免后续积分报错。
    validIdx = ~(isnan(recallCurve) | isnan(precisionCurve));
    recallCurve = recallCurve(validIdx);
    precisionCurve = precisionCurve(validIdx);

    metrics.PRAUC = NaN;
    if numel(recallCurve) > 1 && numel(precisionCurve) > 1
        % 为了积分稳定，先按 Recall 升序排序。
        [recallCurve, orderIdx] = sort(recallCurve);
        precisionCurve = precisionCurve(orderIdx);

        % PR-AUC：PR 曲线下面积。
        metrics.PRAUC = trapz(recallCurve, precisionCurve);

        % AP：Average Precision，常用于不平衡分类评价。
        metrics.AP = computeAveragePrecision(recallCurve, precisionCurve);
    else
        metrics.AP = NaN;
    end
end

function metrics = computeConfusionMetrics(yTrue, yPred)
    % 该函数根据真实标签和预测标签，计算基于混淆矩阵的指标。

    tp = sum((yTrue == 1) & (yPred == 1));
    tn = sum((yTrue == 0) & (yPred == 0));
    fp = sum((yTrue == 0) & (yPred == 1));
    fn = sum((yTrue == 1) & (yPred == 0));

    metrics = struct();
    metrics.Accuracy = (tp + tn) / max(tp + tn + fp + fn, 1);
    metrics.Precision = tp / max(tp + fp, 1);
    metrics.Recall = tp / max(tp + fn, 1);
    metrics.Specificity = tn / max(tn + fp, 1);

    % F1 同时兼顾精确率和召回率。
    metrics.F1 = 2 * metrics.Precision * metrics.Recall / ...
        max(metrics.Precision + metrics.Recall, eps);

    % F2 更强调召回率，在故障检测场景中常有意义。
    metrics.F2 = 5 * metrics.Precision * metrics.Recall / ...
        max(4 * metrics.Precision + metrics.Recall, eps);

    % Balanced Accuracy 用于缓解类别不平衡的影响。
    metrics.BalancedAccuracy = 0.5 * (metrics.Recall + metrics.Specificity);

    % G-Mean 衡量正类与负类识别能力的平衡。
    metrics.GMean = sqrt(max(metrics.Recall * metrics.Specificity, 0));

    % MCC 是不平衡分类中较稳健的综合指标。
    denom = sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), eps));
    metrics.MCC = ((tp * tn) - (fp * fn)) / denom;
end

function ap = computeAveragePrecision(recallCurve, precisionCurve)
    % 根据 PR 曲线近似计算 Average Precision。
    if isempty(recallCurve) || isempty(precisionCurve)
        ap = NaN;
        return;
    end

    recallCurve = recallCurve(:);
    precisionCurve = precisionCurve(:);
    recallDiff = [recallCurve(1); diff(recallCurve)];
    recallDiff(recallDiff < 0) = 0;
    ap = sum(recallDiff .* precisionCurve);
end

function writeRunNotes(config, featureNames, rawTbl, summaryResults)
    % 将本次实验的基本信息写入文本文件，便于回顾和记录。
    noteFile = fullfile(config.outputDir, 'run_notes.txt');
    fid = fopen(noteFile, 'w');
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, 'Predictive maintenance experiment notes\n');
    fprintf(fid, 'Mode: %s\n', config.mode);
    fprintf(fid, 'Folds: %d\n', config.numFolds);
    fprintf(fid, 'Single thread: yes\n');
    fprintf(fid, 'Data file: %s\n\n', config.dataFile);

    fprintf(fid, 'Dataset size: %d rows\n', height(rawTbl));
    positiveCount = sum(rawTbl.("Machine failure") == 1);
    fprintf(fid, 'Positive samples: %d\n', positiveCount);
    fprintf(fid, 'Positive ratio: %.4f\n\n', positiveCount / height(rawTbl));

    fprintf(fid, 'Features used:\n');
    for idx = 1:numel(featureNames)
        fprintf(fid, '- %s\n', featureNames(idx));
    end

    fprintf(fid, '\nBest models ranked by mean F1:\n');
    for idx = 1:height(summaryResults)
        fprintf(fid, ['%d. %s | F1=%.4f | F2=%.4f | Recall=%.4f | MCC=%.4f | ' ...
            'ROC-AUC=%.4f | PR-AUC=%.4f | AP=%.4f\n'], ...
            idx, summaryResults.Model(idx), summaryResults.mean_F1(idx), ...
            summaryResults.mean_F2(idx), summaryResults.mean_Recall(idx), ...
            summaryResults.mean_MCC(idx), summaryResults.mean_ROCAUC(idx), ...
            summaryResults.mean_PRAUC(idx), summaryResults.mean_AP(idx));
    end
end

function plotMetricComparison(summaryResults, outputDir, config)
    % 绘制基础指标对比图：
    % 左图是 F1、Recall、Balanced Accuracy；
    % 右图是 ROC-AUC 和 PR-AUC。

    fig = createFigureWindow(config, [100 100 960 420], '四种算法主要指标对比');
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    bar(categorical(summaryResults.Model), ...
        [summaryResults.mean_F1, summaryResults.mean_Recall, ...
        summaryResults.mean_BalancedAccuracy], 'grouped');
    ylabel('Score');
    title('Classification Metrics');
    legend({'F1', 'Recall', 'Balanced Accuracy'}, 'Location', 'southoutside', ...
        'Orientation', 'horizontal');
    ylim([0 1]);
    grid on;

    nexttile;
    bar(categorical(summaryResults.Model), ...
        [summaryResults.mean_ROCAUC, summaryResults.mean_PRAUC], 'grouped');
    ylabel('AUC');
    title('Ranking Metrics');
    legend({'ROC-AUC', 'PR-AUC'}, 'Location', 'southoutside', ...
        'Orientation', 'horizontal');
    ylim([0 1]);
    grid on;

    finalizeFigureWindow(fig, fullfile(outputDir, 'metric_comparison.png'), config);
end

function plotErrorCurve(foldResults, outputDir, config)
    % 绘制各模型在不同折上的分类错误率曲线。
    fig = createFigureWindow(config, [100 100 840 420], '四种算法误差曲线');
    modelNames = unique(foldResults.Model, 'stable');
    colors = [0.18 0.48 0.8; 0.85 0.33 0.1; 0.25 0.6 0.25; 0.55 0.35 0.75];

    hold on;
    for idx = 1:numel(modelNames)
        name = modelNames(idx);
        rows = foldResults.Model == name;
        folds = foldResults.Fold(rows);
        errorRate = 1 - foldResults.Accuracy(rows);
        plot(folds, errorRate, '-o', 'LineWidth', 2, 'MarkerSize', 6, ...
            'Color', colors(idx, :), 'DisplayName', char(name));
    end

    xlabel('Fold');
    ylabel('Classification Error');
    title('Error Curve of Four Algorithms');
    xlim([1, max(foldResults.Fold)]);
    ylim([0, max(0.25, max(1 - foldResults.Accuracy) + 0.02)]);
    grid on;
    legend('Location', 'eastoutside');

    finalizeFigureWindow(fig, fullfile(outputDir, 'error_curve.png'), config);
end

function plotDatasetOverview(rawTbl, outputDir, config)
    % 绘制数据集概况图：
    % 左边是故障/正常样本数量，右边是产品类型分布。

    fig = createFigureWindow(config, [100 100 980 420], '数据集概览');
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    failureCounts = [sum(rawTbl.("Machine failure") == 0), ...
        sum(rawTbl.("Machine failure") == 1)];
    b1 = bar(categorical({'Normal', 'Failure'}), failureCounts, 0.55);
    b1.FaceColor = 'flat';
    b1.CData = [0.2 0.45 0.8; 0.85 0.33 0.1];
    ylabel('Samples');
    title('Class Distribution');
    grid on;

    nexttile;
    typeCats = categories(categorical(rawTbl.Type));
    typeCounts = zeros(numel(typeCats), 1);
    for idx = 1:numel(typeCats)
        typeCounts(idx) = sum(categorical(rawTbl.Type) == categorical(typeCats(idx)));
    end
    b2 = bar(categorical(typeCats), typeCounts, 0.55);
    b2.FaceColor = 'flat';
    b2.CData = [0.25 0.6 0.85; 0.5 0.7 0.35; 0.85 0.65 0.13];
    ylabel('Samples');
    title('Product Type Distribution');
    grid on;

    finalizeFigureWindow(fig, fullfile(outputDir, 'dataset_overview.png'), config);
end

function analysisMap = analyzeAllModels(featureTbl, featureMatrix, target, featureNames, config)
    % 该函数在一个固定的保持集上分析所有模型，
    % 以便统一生成 ROC 曲线、PR 曲线、混淆矩阵等图。

    holdout = cvpartition(target, 'HoldOut', 0.2);
    trainIdx = training(holdout);
    testIdx = test(holdout);

    XTrainTable = featureTbl(trainIdx, :);
    XTestTable = featureTbl(testIdx, :);
    XTrainMatrix = featureMatrix(trainIdx, :);
    XTestMatrix = featureMatrix(testIdx, :);
    yTrain = target(trainIdx);
    yTest = target(testIdx);

    obsWeights = makePositiveClassWeights(yTrain);
    models = trainModels(XTrainTable, XTrainMatrix, yTrain, obsWeights, config);
    analysisMap = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for idx = 1:numel(models)
        modelObject = models(idx).Model;
        inputKind = models(idx).InputKind;
        modelName = string(models(idx).Name);
        [predLabel, positiveScore] = predictWithScore(modelObject, inputKind, ...
            XTestTable, XTestMatrix);

        % 将每个模型的评估结果都保存成结构体，后续统一画图。
        analysis = struct();
        analysis.ModelName = modelName;
        analysis.ModelObject = modelObject;
        analysis.InputKind = inputKind;
        analysis.FeatureNames = featureNames;
        analysis.TargetTrue = yTest;
        analysis.TargetPred = predLabel;
        analysis.PositiveScore = positiveScore;
        analysis.Metrics = computeBinaryMetrics(yTest, predLabel, positiveScore);

        % 记录正类比例，用于 PR 曲线中的基线显示。
        analysis.PositiveRate = mean(double(yTest) - 1);

        % 保存混淆矩阵。
        analysis.ConfusionMatrix = confusionmat(double(yTest) - 1, ...
            double(predLabel) - 1, 'Order', [0 1]);

        % 保存 ROC 曲线点。
        [analysis.RocX, analysis.RocY, ~, analysis.Metrics.ROCAUC] = ...
            perfcurve(double(yTest) - 1, positiveScore, 1);

        % 保存 PR 曲线点。
        [analysis.PrRecall, analysis.PrPrecision] = perfcurve(double(yTest) - 1, ...
            positiveScore, 1, 'xCrit', 'reca', 'yCrit', 'prec');

        validIdx = ~(isnan(analysis.PrRecall) | isnan(analysis.PrPrecision));
        analysis.PrRecall = analysis.PrRecall(validIdx);
        analysis.PrPrecision = analysis.PrPrecision(validIdx);

        % 保存阈值扫描结果，用于后续阈值分析图。
        analysis.ThresholdSweep = computeThresholdSweep(double(yTest) - 1, positiveScore);

        % 尝试提取特征重要性；若模型不支持，则返回 NaN。
        try
            analysis.Importance = predictorImportance(modelObject);
        catch
            analysis.Importance = nan(1, numel(featureNames));
        end

        analysisMap(char(modelName)) = analysis;
    end
end

function saveAllModelHoldoutMetrics(analysisMap, outputDir)
    % 将所有模型在保持集上的结果保存成表格。
    keysList = ["Logistic", "DecisionTree", "BaggedTrees", "RUSBoost"];
    rows = table();

    for idx = 1:numel(keysList)
        analysis = analysisMap(char(keysList(idx)));
        m = analysis.Metrics;
        row = table(string(analysis.ModelName), m.Accuracy, m.Precision, m.Recall, ...
            m.Specificity, m.F1, m.F2, m.GMean, m.MCC, m.ROCAUC, m.PRAUC, ...
            m.AP, m.BalancedAccuracy, ...
            'VariableNames', {'Model', 'Accuracy', 'Precision', 'Recall', ...
            'Specificity', 'F1', 'F2', 'GMean', 'MCC', 'ROCAUC', 'PRAUC', ...
            'AP', 'BalancedAccuracy'});
        rows = [rows; row]; %#ok<AGROW>
    end

    writetable(rows, fullfile(outputDir, 'holdout_metrics.csv'));
end

function plotRankingCurvesComparison(analysisMap, outputDir, config)
    % 同时绘制四种算法的 ROC 曲线和 PR 曲线，便于横向对比。

    orderedNames = ["Logistic", "DecisionTree", "BaggedTrees", "RUSBoost"];
    colors = [0.18 0.48 0.8; 0.85 0.33 0.1; 0.25 0.6 0.25; 0.55 0.35 0.75];
    fig = createFigureWindow(config, [100 100 980 430], '四种算法 ROC 与 PR 曲线');
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    % 左侧：ROC 曲线。
    nexttile;
    hold on;
    for idx = 1:numel(orderedNames)
        analysis = analysisMap(char(orderedNames(idx)));
        plot(analysis.RocX, analysis.RocY, 'LineWidth', 2, 'Color', colors(idx, :), ...
            'DisplayName', sprintf('%s (AUC=%.3f)', orderedNames(idx), ...
            analysis.Metrics.ROCAUC));
    end
    plot([0 1], [0 1], '--', 'Color', [0.6 0.6 0.6], 'DisplayName', 'Random');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('ROC Curves of Four Algorithms');
    grid on;
    legend('Location', 'southoutside', 'Orientation', 'vertical');

    % 右侧：PR 曲线。
    nexttile;
    hold on;
    prevalence = analysisMap(char(orderedNames(1))).PositiveRate;
    for idx = 1:numel(orderedNames)
        analysis = analysisMap(char(orderedNames(idx)));
        plot(analysis.PrRecall, analysis.PrPrecision, 'LineWidth', 2, ...
            'Color', colors(idx, :), 'DisplayName', sprintf('%s (AP=%.3f)', ...
            orderedNames(idx), analysis.Metrics.AP));
    end
    yline(prevalence, '--', 'Baseline', 'Color', [0.55 0.55 0.55], ...
        'LineWidth', 1.2, 'DisplayName', sprintf('Baseline=%.3f', prevalence));
    xlabel('Recall');
    ylabel('Precision');
    title('Precision-Recall Curves of Four Algorithms');
    xlim([0 1]);
    ylim([0 max(0.9, prevalence + 0.05)]);
    grid on;
    legend('Location', 'southoutside', 'Orientation', 'vertical');

    finalizeFigureWindow(fig, fullfile(outputDir, 'all_model_rank_curves.png'), config);
end

function plotAdvancedMetricHeatmap(summaryResults, outputDir, config)
    % 绘制增强指标热力图，
    % 用颜色深浅展示不同模型在多项指标上的强弱。

    orderedNames = ["Logistic", "DecisionTree", "BaggedTrees", "RUSBoost"];
    orderedRows = zeros(numel(orderedNames), 1);
    for idx = 1:numel(orderedNames)
        orderedRows(idx) = find(summaryResults.Model == orderedNames(idx), 1);
    end
    orderedSummary = summaryResults(orderedRows, :);

    metricMatrix = [orderedSummary.mean_Recall, orderedSummary.mean_F1, ...
        orderedSummary.mean_F2, orderedSummary.mean_GMean, ...
        orderedSummary.mean_BalancedAccuracy, orderedSummary.mean_MCC, ...
        orderedSummary.mean_ROCAUC, orderedSummary.mean_PRAUC, ...
        orderedSummary.mean_AP];
    metricNames = {'Recall', 'F1', 'F2', 'G-Mean', 'Balanced Acc', ...
        'MCC', 'ROC-AUC', 'PR-AUC', 'AP'};

    fig = createFigureWindow(config, [100 100 980 360], '四种算法增强指标热力图');
    cmap = [0.96 0.98 1.00;
        0.84 0.91 0.97;
        0.67 0.80 0.91;
        0.45 0.64 0.82;
        0.23 0.43 0.69;
        0.08 0.24 0.49];
    imagesc(metricMatrix);
    colormap(cmap);
    colorbar;
    title('Advanced Metric Heatmap of Four Algorithms');
    xticks(1:numel(metricNames));
    xticklabels(metricNames);
    yticks(1:numel(orderedNames));
    yticklabels(cellstr(orderedNames));

    ax = gca;
    ax.XTickLabelRotation = 25;

    % 根据数值大小自动决定文字颜色，避免看不清。
    minVal = min(metricMatrix(:));
    maxVal = max(metricMatrix(:));
    spanVal = max(maxVal - minVal, eps);
    for r = 1:size(metricMatrix, 1)
        for c = 1:size(metricMatrix, 2)
            val = metricMatrix(r, c);
            normVal = (val - minVal) / spanVal;
            if normVal > 0.55
                textColor = 'w';
            else
                textColor = 'k';
            end
            text(c, r, sprintf('%.3f', val), 'HorizontalAlignment', 'center', ...
                'Color', textColor, 'FontWeight', 'bold', 'FontSize', 10);
        end
    end

    finalizeFigureWindow(fig, fullfile(outputDir, 'advanced_metric_heatmap.png'), config);
end

function sweep = computeThresholdSweep(yTrue, positiveScore)
    % 该函数扫描不同分类阈值下的表现，
    % 用于观察 Precision、Recall、F1、F2、MCC 和 G-Mean 的变化趋势。

    if all(positiveScore >= 0 & positiveScore <= 1)
        thresholds = linspace(0, 1, 101)';
    else
        thresholds = linspace(min(positiveScore), max(positiveScore), 101)';
    end

    precision = zeros(size(thresholds));
    recall = zeros(size(thresholds));
    f1 = zeros(size(thresholds));
    f2 = zeros(size(thresholds));
    mcc = zeros(size(thresholds));
    gmean = zeros(size(thresholds));

    for idx = 1:numel(thresholds)
        yPred = double(positiveScore >= thresholds(idx));
        m = computeConfusionMetrics(yTrue, yPred);
        precision(idx) = m.Precision;
        recall(idx) = m.Recall;
        f1(idx) = m.F1;
        f2(idx) = m.F2;
        mcc(idx) = m.MCC;
        gmean(idx) = m.GMean;
    end

    sweep = struct();
    sweep.Thresholds = thresholds;
    sweep.Precision = precision;
    sweep.Recall = recall;
    sweep.F1 = f1;
    sweep.F2 = f2;
    sweep.MCC = mcc;
    sweep.GMean = gmean;
end

function plotAllModelConfusions(analysisMap, outputDir, config)
    % 将四种模型的混淆矩阵画在同一张图里，方便横向比较。

    orderedNames = ["Logistic", "DecisionTree", "BaggedTrees", "RUSBoost"];
    fig = createFigureWindow(config, [100 100 900 760], '四种算法混淆矩阵');
    tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    cmap = [0.95 0.97 1.00;
        0.78 0.86 0.95;
        0.55 0.72 0.89;
        0.30 0.53 0.78;
        0.10 0.28 0.52];

    for idx = 1:numel(orderedNames)
        analysis = analysisMap(char(orderedNames(idx)));
        cm = analysis.ConfusionMatrix;

        nexttile;
        imagesc(cm);
        axis image;
        colormap(cmap);
        colorbar;
        title(char(orderedNames(idx)));
        xlabel('Predicted Class');
        ylabel('True Class');
        xticks([1 2]);
        yticks([1 2]);
        xticklabels({'0', '1'});
        yticklabels({'0', '1'});

        % 根据背景深浅自动调整格子里的数字颜色。
        maxVal = max(cm(:));
        for r = 1:2
            for c = 1:2
                if cm(r, c) > 0.55 * maxVal
                    textColor = 'w';
                else
                    textColor = 'k';
                end
                text(c, r, num2str(cm(r, c)), 'HorizontalAlignment', 'center', ...
                    'Color', textColor, 'FontWeight', 'bold', 'FontSize', 12);
            end
        end
    end

    finalizeFigureWindow(fig, fullfile(outputDir, 'all_model_confusions.png'), config);
end

function writeBestModelNotes(analysis, outputDir)
    % 将最佳模型的详细结果单独写入文本文件
    noteFile = fullfile(outputDir, 'best_model_summary.txt');
    fid = fopen(noteFile, 'w');
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, 'Best model diagnostic summary\n');
    fprintf(fid, 'Model: %s\n', analysis.ModelName);
    fprintf(fid, 'Accuracy: %.4f\n', analysis.Metrics.Accuracy);
    fprintf(fid, 'Precision: %.4f\n', analysis.Metrics.Precision);
    fprintf(fid, 'Recall: %.4f\n', analysis.Metrics.Recall);
    fprintf(fid, 'Specificity: %.4f\n', analysis.Metrics.Specificity);
    fprintf(fid, 'F1: %.4f\n', analysis.Metrics.F1);
    fprintf(fid, 'F2: %.4f\n', analysis.Metrics.F2);
    fprintf(fid, 'G-Mean: %.4f\n', analysis.Metrics.GMean);
    fprintf(fid, 'MCC: %.4f\n', analysis.Metrics.MCC);
    fprintf(fid, 'ROC-AUC: %.4f\n', analysis.Metrics.ROCAUC);
    fprintf(fid, 'PR-AUC: %.4f\n', analysis.Metrics.PRAUC);
    fprintf(fid, 'AP: %.4f\n', analysis.Metrics.AP);
    fprintf(fid, 'Balanced Accuracy: %.4f\n\n', analysis.Metrics.BalancedAccuracy);

    cm = analysis.ConfusionMatrix;
    fprintf(fid, 'Confusion matrix [[TN, FP], [FN, TP]] = [[%d, %d], [%d, %d]]\n\n', ...
        cm(1, 1), cm(1, 2), cm(2, 1), cm(2, 2));

    % 若模型支持特征重要性，则输出前 5 个最重要特征。
    if all(~isnan(analysis.Importance))
        [sortedImportance, sortIdx] = sort(analysis.Importance, 'descend');
        fprintf(fid, 'Top feature importance:\n');
        for idx = 1:min(5, numel(sortIdx))
            fprintf(fid, '%d. %s = %.4f\n', idx, analysis.FeatureNames(sortIdx(idx)), ...
                sortedImportance(idx));
        end
    end
end

function plotBestModelDiagnostics(analysis, outputDir, config)
    % 统一调用最佳模型的所有诊断图函数。
    plotBestModelCurves(analysis, outputDir, config);
    plotBestModelConfusion(analysis, outputDir, config);
    plotBestModelImportance(analysis, outputDir, config);
    plotBestModelThresholdMetrics(analysis, outputDir, config);
end

function plotBestModelCurves(analysis, outputDir, config)
    % 绘制最佳模型的 ROC 曲线和 PR 曲线。
    fig = createFigureWindow(config, [100 100 980 420], '最佳模型 ROC 与 PR 曲线');
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    plot(analysis.RocX, analysis.RocY, 'LineWidth', 2, 'Color', [0.18 0.48 0.8]);
    hold on;
    plot([0 1], [0 1], '--', 'Color', [0.5 0.5 0.5]);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(sprintf('%s ROC Curve (AUC = %.3f)', analysis.ModelName, ...
        analysis.Metrics.ROCAUC));
    grid on;

    nexttile;
    plot(analysis.PrRecall, analysis.PrPrecision, 'LineWidth', 2, ...
        'Color', [0.85 0.33 0.1]);
    xlabel('Recall');
    ylabel('Precision');
    title(sprintf('%s PR Curve (AUC = %.3f)', analysis.ModelName, ...
        analysis.Metrics.PRAUC));
    xlim([0 1]);
    ylim([0 0.9]);
    grid on;

    finalizeFigureWindow(fig, fullfile(outputDir, 'best_model_curves.png'), config);
end

function plotBestModelThresholdMetrics(analysis, outputDir, config)
    % 绘制最佳模型在不同分类阈值下的指标变化，
    % 用于观察 Precision 和 Recall 等指标之间的权衡。

    sweep = analysis.ThresholdSweep;
    fig = createFigureWindow(config, [100 100 980 430], '最佳模型阈值分析');
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    % 左图：Precision / Recall / F1 / F2 随阈值变化。
    nexttile;
    hold on;
    plot(sweep.Thresholds, sweep.Precision, 'LineWidth', 2, 'Color', [0.85 0.33 0.1]);
    plot(sweep.Thresholds, sweep.Recall, 'LineWidth', 2, 'Color', [0.18 0.48 0.8]);
    plot(sweep.Thresholds, sweep.F1, 'LineWidth', 2, 'Color', [0.25 0.6 0.25]);
    plot(sweep.Thresholds, sweep.F2, 'LineWidth', 2, 'Color', [0.55 0.35 0.75]);
    xlabel('Decision Threshold');
    ylabel('Score');
    title(sprintf('%s Precision/Recall/F1/F2 vs Threshold', analysis.ModelName));
    ylim([0 1]);
    grid on;
    legend({'Precision', 'Recall', 'F1', 'F2'}, 'Location', 'southoutside', ...
        'Orientation', 'horizontal');

    % 右图：MCC / G-Mean 随阈值变化。
    nexttile;
    hold on;
    plot(sweep.Thresholds, sweep.MCC, 'LineWidth', 2, 'Color', [0.2 0.2 0.2]);
    plot(sweep.Thresholds, sweep.GMean, 'LineWidth', 2, 'Color', [0.10 0.55 0.70]);
    xlabel('Decision Threshold');
    ylabel('Score');
    title(sprintf('%s MCC/G-Mean vs Threshold', analysis.ModelName));
    ylim([0 1]);
    grid on;
    legend({'MCC', 'G-Mean'}, 'Location', 'southoutside', ...
        'Orientation', 'horizontal');

    finalizeFigureWindow(fig, fullfile(outputDir, 'best_model_threshold_metrics.png'), config);
end

function plotBestModelConfusion(analysis, outputDir, config)
    % 绘制最佳模型的单独混淆矩阵图。
    cm = analysis.ConfusionMatrix;
    fig = createFigureWindow(config, [100 100 500 420], '最佳模型混淆矩阵');
    cmap = [0.95 0.97 1.00;
        0.78 0.86 0.95;
        0.55 0.72 0.89;
        0.30 0.53 0.78;
        0.10 0.28 0.52];

    imagesc(cm);
    axis image;
    colormap(cmap);
    colorbar;
    title(sprintf('%s Confusion Matrix', analysis.ModelName));
    xlabel('Predicted Class');
    ylabel('True Class');
    xticks([1 2]);
    yticks([1 2]);
    xticklabels({'Pred 0', 'Pred 1'});
    yticklabels({'True 0', 'True 1'});

    % 自动调整数字颜色
    maxVal = max(cm(:));
    for r = 1:2
        for c = 1:2
            if cm(r, c) > 0.55 * maxVal
                textColor = 'w';
            else
                textColor = 'k';
            end
            text(c, r, num2str(cm(r, c)), 'HorizontalAlignment', 'center', ...
                'Color', textColor, 'FontWeight', 'bold', 'FontSize', 13);
        end
    end

    finalizeFigureWindow(fig, fullfile(outputDir, 'best_model_confusion.png'), config);
end

function plotBestModelImportance(analysis, outputDir, config)
    % 若模型支持特征重要性，则绘制前若干个最重要特征。
    if all(isnan(analysis.Importance))
        return;
    end

    [sortedImportance, sortIdx] = sort(analysis.Importance, 'descend');
    topK = min(8, numel(sortIdx));
    sortedImportance = sortedImportance(1:topK);
    sortedNames = analysis.FeatureNames(sortIdx(1:topK));

    fig = createFigureWindow(config, [100 100 780 420], '最佳模型特征重要性');
    barh(categorical(sortedNames(end:-1:1)), sortedImportance(end:-1:1), 0.6, ...
        'FaceColor', [0.22 0.56 0.25]);
    xlabel('Importance');
    ylabel('Feature');
    title(sprintf('Top Feature Importance of %s', analysis.ModelName));
    grid on;

    finalizeFigureWindow(fig, fullfile(outputDir, 'best_model_importance.png'), config);
end

function fig = createFigureWindow(config, positionVec, figName)
    
    if isfield(config, 'showFigures') && config.showFigures
        visibleState = 'on';
    else
        visibleState = 'off';
    end

    fig = figure('Visible', visibleState, ...
        'Color', 'w', ...
        'Position', positionVec, ...
        'Name', figName, ...
        'NumberTitle', 'off');

    if strcmp(visibleState, 'on')
        drawnow;
    end
end

function finalizeFigureWindow(fig, outputPath, config)
    % 保存图片；
    exportgraphics(fig, outputPath, 'Resolution', 150);

    if ~(isfield(config, 'showFigures') && config.showFigures)
        close(fig);
    else
        drawnow;
    end
end
