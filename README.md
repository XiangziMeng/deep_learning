flowchart TD
    %% 数据源层
    subgraph A [数据源层]
        A1[题目表]
        A2[学生表]
        A3[题模表]
        A4[场景配置表]
        A5[交互日志流]
    end

    %% 特征工程层
    subgraph B [特征工程与计算层]
        B1[原子特征]
        
        subgraph B1_1 [ ]
            B1a[题目特征<br>难度、知识点等]
            B1b[学生特征<br>能力值、历史正确率等]
            B1c[题模特征<br>题目数量、策略等]
            B1d[场景特征<br>类型、是否限时等]
        end

        B2[交叉特征<br>e.g., 难度与能力匹配度]
        B3[(特征存储)]
    end

    %% 服务应用层
    subgraph C [服务与应用层]
        C1[离线训练<br>生成训练数据]
        C2[在线推理<br>模型预测服务]
    end

    %% 数据流
    A1 --> B1a
    A2 --> B1b
    A3 --> B1c
    A4 --> B1d
    A5 --> B1b

    B1a --> B2
    B1b --> B2
    B1c --> B2
    B1d --> B2

    B1a --> B3
    B1b --> B3
    B1c --> B3
    B1d --> B3
    B2 --> B3

    B3 --> C1
    B3 --> C2
