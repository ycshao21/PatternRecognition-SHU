import pandas as pd

def Read() -> pd.DataFrame:
    '''
    Format:
          性别  |  身高(cm) | 体重(kg) | 脚长(mm) | 尺码
        ----------------------------------------------------
          (str) | (float)  |  (int)   | (float)  | (float)
    '''
    data1 = pd.read_excel("Dataset/2022冬模式识别数据收集.xlsx", index_col=0)
    data2 = pd.read_excel("Dataset/2021冬模式识别数据收集.xlsx", index_col=0)
    data1 = data1.iloc[:, :5]
    data2 = data2.iloc[:, :5]
    data = pd.concat([data1, data2])
    return data