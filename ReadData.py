import pandas as pd
from Utils.Logger import Logger

def ReadGenderData() -> pd.DataFrame:
    data = []

    ####################################################################################
    # 2009
    ####################################################################################
    Logger.Log("Reading 2009 male data...")
    data2009_Male = pd.read_csv("Dataset/genderdata/boy2009.txt", names=['身高(cm)', '体重(kg)', '鞋码'], sep=r'\s+')
    data2009_Male['性别'] = '男'
    data.append(data2009_Male)

    Logger.Log("Reading 2009 female data...")
    data2009_Female = pd.read_csv("Dataset/genderdata/girl2009.txt", names=['身高(cm)', '体重(kg)', '鞋码'], sep=r'\s+')
    data2009_Female['性别'] = '女'
    data.append(data2009_Female)

    ####################################################################################
    # 2010
    ####################################################################################
    Logger.Log("Reading 2010 male data...")
    data2010_Male = pd.read_csv("Dataset/genderdata/boy2010.txt", names=['身高(cm)', '体重(kg)', '鞋码'], sep=r'\s+')
    data2010_Male['性别'] = '男'
    data.append(data2010_Male)

    Logger.Log("Reading 2010 female data...")
    data2010_Female = pd.read_csv("Dataset/genderdata/girl2010.txt", names=['身高(cm)', '体重(kg)', '鞋码'], sep=r'\s+')
    data2010_Female['性别'] = '女'
    data.append(data2010_Female)

    ####################################################################################
    # 2011
    ####################################################################################
    Logger.Log("Reading 2011 male data...")
    data2011_Male = pd.read_csv("Dataset/genderdata/boy2011.txt", names=['身高(cm)', '体重(kg)', '鞋码'], sep=r'\s+')
    data2011_Male['性别'] = '男'
    data.append(data2011_Male)

    Logger.Log("Reading 2011 female data...")
    data2011_Female = pd.read_csv("Dataset/genderdata/girl2011.txt", names=['身高(cm)', '体重(kg)', '鞋码'], sep=r'\s+')
    data2011_Female['性别'] = '女'
    data.append(data2011_Female)

    ####################################################################################
    # 2017
    ####################################################################################
    Logger.Log("Reading 2017 male data...")
    data2017_Male = pd.read_csv("Dataset/genderdata/boy2017.txt", names=['身高(cm)', '体重(kg)', '鞋码'], skiprows=[0], sep=r'\s+')
    data2017_Male['性别'] = '男'
    data.append(data2017_Male)

    Logger.Log("Reading 2017 female data...")
    data2017_Female = pd.read_csv("Dataset/genderdata/girl2017.txt", names=['身高(cm)', '体重(kg)', '鞋码'], skiprows=[0], sep=r'\s+')
    data2017_Female['性别'] = '女'
    data.append(data2017_Female)

    ####################################################################################
    # 2018
    ####################################################################################
    Logger.Log("Reading 2018 data...")
    data2018 = pd.read_csv("Dataset/genderdata/boyandgirl2018.csv")
    data2018.rename(columns={'身高（cm）':'身高(cm)', '体重（kg）':'体重(kg)'}, inplace=True)
    data.append(data2018)

    ####################################################################################
    # 2021
    ####################################################################################
    Logger.Log("Reading 2021 data...")
    data2021 = pd.read_csv("Dataset/genderdata/2021冬模式识别数据收集.csv", index_col=0)
    data2021 = data2021.iloc[:, :5]
    data2021.rename(columns={'尺码': '鞋码'}, inplace=True)
    data.append(data2021)

    ####################################################################################
    # 2022
    ####################################################################################
    Logger.Log("Reading 2022 data...")
    data2022 = pd.read_csv("Dataset/genderdata/2022冬模式识别数据收集.csv", index_col=0)
    data2022 = data2022.iloc[:, :5]
    data2022.rename(columns={'尺码': '鞋码'}, inplace=True)
    data.append(data2022)

    # Merge data
    data = pd.concat(data, ignore_index=True)
    Logger.Log(f"Data merged. {len(data)} rows in total.")

    # Clean data
    nanData = data[['身高(cm)', '体重(kg)', '鞋码', '性别']].isnull().any(axis=1)
    data.dropna(subset=['身高(cm)', '体重(kg)', '鞋码', '性别'], inplace=True)
    Logger.Log(f"Dropped {nanData.sum()} rows with NaN values. Left {len(data)} rows.")

    data.loc[data['性别'] == '男', '性别'] = 1
    data.loc[data['性别'] == '女', '性别'] = 0

    return data


if __name__ == '__main__':
    data = ReadGenderData()
    savePath = "Dataset/genderData_Merged.csv"
    data.to_csv(savePath, index=False, encoding='utf-8')
    Logger.Log(f"Saved to {savePath}.")
