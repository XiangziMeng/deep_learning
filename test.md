Here is a simple flow chart:

```mermaid
flowchart TD
    %% 数据源层
    subgraph A [数据预处理]
        A1[原始数据]
        A2[MAD数据清洗]
        B1[id类、类别特征<br>学生id、题目id、题目类别]
        B2[数值特征<br>题目平均时长、学生答题正确率]
        B3[序列特征<br>学生答题题模id序列]
        C1[独热编码]
        C2[归一化<br>均值为0、方差为1]
        C3[独热编码、平均池化]
    end

    %% DNN模型
    subgraph D [DNN模型]
        D1[模型训练<br>loss,ape<0.3占比]
        D2[在线推理<br>ape<0.3占比]
    end

    %% 数据流
    A1 --> A2
    A2 --> B1
    A2 --> B2
    A2 --> B3
    B1 --> C1
    B2 --> C2
    B3 --> C3
    A --> D  
```
