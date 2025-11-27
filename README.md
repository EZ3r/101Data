## LSTM-AE + GMM 异常检测流程

```mermaid
graph TD
    %% 定义样式
    classDef data fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:5,ry:5;
    classDef process fill:#fff3e0,stroke:#e65100,stroke-width:2px,rx:5,ry:5;
    classDef model fill:#dcedc8,stroke:#33691e,stroke-width:2px,rx:15,ry:15;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,rx:5,ry:5;
    classDef output fill:#ffcdd2,stroke:#b71c1c,stroke-width:2px,rx:5,ry:5;

    %% ================= 1. 数据预处理阶段 =================
    subgraph Phase1 [阶段一 通用数据预处理]
        RawData["原始传感器 CSV 数据"] ::: data
        SelectCols["选择相关特征列"] ::: process
        DiffProcess["一阶差分处理 [使数据平稳]"] ::: process
        SplitData["划分训练集 测试集"] ::: process
        
        RawData --> SelectCols
        SelectCols --> DiffProcess
        DiffProcess --> SplitData

        subgraph TrainingPrep [训练集准备]
            FitScaler["MinMaxScaler 拟合 [Fit]"] ::: process
            TransformTrain["归一化转换 [Transform]"] ::: process
            SlidingWindowTrain["滑动时间窗口构建序列 [T=150]"] ::: process
            TrainSeqs["训练集序列数据 [N_train 150 78]"] ::: data
            
            SplitData -- 训练数据 --> FitScaler
            FitScaler --> TransformTrain
            TransformTrain --> SlidingWindowTrain
            SlidingWindowTrain --> TrainSeqs
        end
        
        ScalerModel[("已拟合 Scaler [用于后续转换]")] ::: model
        FitScaler -. 保存 Scaler 参数 .-> ScalerModel
    end

    %% ================= 2. LSTM-AE 训练阶段 =================
    subgraph Phase2 [阶段二 LSTM-AE 模型训练 - 特征提取器学习]
        DefineAE["构建 LSTM-AE 网络结构 [Encoder-Decoder]"] ::: process
        TrainAE["模型训练 [目标函数: 最小化 MSE 重构误差]"] ::: process
        TrainedAE[("已训练 LSTM-AE 整体模型")] ::: model
        ExtractEncoder["提取 Encoder 部分 [作为特征提取器]"] ::: process
        TrainedEncoder[("已提取 Encoder 模型")] ::: model
        
        TrainSeqs --> DefineAE
        DefineAE --> TrainAE
        TrainAE -- 训练完成 --> TrainedAE
        TrainedAE --> ExtractEncoder
        ExtractEncoder --> TrainedEncoder
    end

    %% ================= 3. GMM 训练与阈值设定阶段 =================
    subgraph Phase3 [阶段三 GMM 建模与阈值设定]
        EncoderInferTrain["使用 Encoder 进行推理 降维"] ::: process
        LatentTrainVecs["训练集潜在向量 [N_train Latent_Dim]"] ::: data
        InitGMM["初始化 GMM"] ::: process
        FitGMM["GMM 模型拟合 [学习潜在空间的正常概率分布]"] ::: process
        TrainedGMM[("已训练 GMM 模型")] ::: model
        
        TrainSeqs -- 输入训练序列 --> EncoderInferTrain
        TrainedEncoder -. 提供参数 .-> EncoderInferTrain
        EncoderInferTrain -- 输出 --> LatentTrainVecs
        LatentTrainVecs --> InitGMM
        InitGMM --> FitGMM
        FitGMM -- 训练完成 --> TrainedGMM

        subgraph Thresholding [阈值确定]
            ScoreTrain["计算训练集 Log-Likelihood 分数"] ::: process
            TrainScores["训练集分数分布"] ::: data
            CalcStats["计算均值 Mean 和标准差 Std"] ::: process
            SetThreshold["设定异常阈值 [例如: Mean - 3*Std]"] ::: process
            FinalThreshold[("最终异常阈值 [Threshold]")] ::: output
            
            LatentTrainVecs --> ScoreTrain
            TrainedGMM -. 提供分布参数 .-> ScoreTrain
            ScoreTrain --> TrainScores
            TrainScores --> CalcStats
            CalcStats --> SetThreshold
            SetThreshold --> FinalThreshold
        end
    end

    %% ================= 4. 在线推理/检测流程 =================
    subgraph Inference [阶段四 在线检测推理流程 - 新数据]
        NewRawData["新采集传感器数据"] ::: data
        NewDiff["一阶差分处理"] ::: process
        NewTransform["归一化转换 [使用已保存的 Scaler]"] ::: process
        NewSlidingWindow["滑动时间窗口构建新序列"] ::: process
        NewSeqs["新测试序列 [N_test 150 78]"] ::: data
        EncoderInferNew["Encoder 推理降维"] ::: process
        NewLatentVecs["新潜在向量 [N_test Latent_Dim]"] ::: data
        GMMScoreNew["GMM 打分 [计算 Log-Likelihood]"] ::: process
        NewScores["新数据健康分数"] ::: data
        CompareThreshold{"分数 < 阈值?"} ::: decision
        AnomalyDetected["判定为异常 故障前兆"] ::: output
        NormalDetected["判定为正常"] ::: data

        NewRawData --> NewDiff
        NewDiff --> NewTransform
        ScalerModel -. 加载参数 .-> NewTransform
        NewTransform --> NewSlidingWindow
        NewSlidingWindow --> NewSeqs
        NewSeqs --> EncoderInferNew
        TrainedEncoder -. 加载模型 .-> EncoderInferNew
        EncoderInferNew --> NewLatentVecs
        NewLatentVecs --> GMMScoreNew
        TrainedGMM -. 加载模型 .-> GMMScoreNew
        GMMScoreNew --> NewScores
        NewScores --> CompareThreshold
        FinalThreshold -. 对比标准 .-> CompareThreshold
        
        CompareThreshold -- "Yes [Log-Likelihood过低]" --> AnomalyDetected
        CompareThreshold -- "No [位于高概率区域]" --> NormalDetected
    end
