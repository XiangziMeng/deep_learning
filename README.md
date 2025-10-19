Here is a simple flow chart:

```mermaid
flowchart TD
    %% 数据源层
    subgraph A [数据源层]
        A1[题目表]
        A2[学生表]
        A3[学生答题记录表]
    end

    %% 特征工程层
    subgraph B [特征工程与计算层]
        B1[题目特征<br>题目平均时长，题目类型等]
        B2[学生特征<br>学生等级，学生因子等]
        B3[题模特征<br>题模平均时长，题模等级等]
        B4[交叉特征<br>题目学生类型平均时长]
        B5[特征存储]
    end

    %% 服务应用层
    subgraph C [服务与应用层]
        C1[离线训练<br>构建训练数据]
        C2[在线推理<br>在线模型预测]
    end

    %% 数据流
    A1 --> B1
    A3 --> B1
    A2 --> B2
    A3 --> B2
    A3 --> B3
    A1 --> B4
    A2 --> B4
    A3 --> B4

    B1 --> B5
    B2 --> B5
    B3 --> B5
    B4 --> B5

    B5 --> C1
    B5 --> C2
```
