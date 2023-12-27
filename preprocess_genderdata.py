import pandas as pd
import logging
import os
import initialize

initialize.init()

logger = logging.getLogger(name="Preprocessing")


def read_genderdata() -> pd.DataFrame:
    """
    Read genderdata

    Female -- 0
    Male -- 1

    Returns
    -------
    pd.DataFrame
        Preprocessed dataset.
    """
    data = []

    # 2009
    logger.info("Reading 2009 male data...")
    data2009_Male = pd.read_csv(
        "dataset/genderdata/raw/boy2009.txt",
        names=["height(cm)", "weight(kg)", "shoe_size"],
        sep=r"\s+",
    )
    data2009_Male["sex"] = 1
    data.append(data2009_Male)

    logger.info("Reading 2009 female data...")
    data2009_Female = pd.read_csv(
        "dataset/genderdata/raw/girl2009.txt",
        names=["height(cm)", "weight(kg)", "shoe_size"],
        sep=r"\s+",
    )
    data2009_Female["sex"] = 0
    data.append(data2009_Female)

    # 2010
    logger.info("Reading 2010 male data...")
    data2010_Male = pd.read_csv(
        "dataset/genderdata/raw/boy2010.txt",
        names=["height(cm)", "weight(kg)", "shoe_size"],
        sep=r"\s+",
    )
    data2010_Male["sex"] = 1
    data.append(data2010_Male)

    logger.info("Reading 2010 female data...")
    data2010_Female = pd.read_csv(
        "dataset/genderdata/raw/girl2010.txt",
        names=["height(cm)", "weight(kg)", "shoe_size"],
        sep=r"\s+",
    )
    data2010_Female["sex"] = 0
    data.append(data2010_Female)

    # 2011
    logger.info("Reading 2011 male data...")
    data2011_Male = pd.read_csv(
        "dataset/genderdata/raw/boy2011.txt",
        names=["height(cm)", "weight(kg)", "shoe_size"],
        sep=r"\s+",
    )
    data2011_Male["sex"] = 1
    data.append(data2011_Male)

    logger.info("Reading 2011 female data...")
    data2011_Female = pd.read_csv(
        "dataset/genderdata/raw/girl2011.txt",
        names=["height(cm)", "weight(kg)", "shoe_size"],
        sep=r"\s+",
    )
    data2011_Female["sex"] = 0
    data.append(data2011_Female)

    # 2017
    logger.info("Reading 2017 male data...")
    data2017_Male = pd.read_csv(
        "dataset/genderdata/raw/boy2017.txt",
        names=["height(cm)", "weight(kg)", "shoe_size"],
        skiprows=[0],
        sep=r"\s+",
    )
    data2017_Male["sex"] = 1
    data.append(data2017_Male)

    logger.info("Reading 2017 female data...")
    data2017_Female = pd.read_csv(
        "dataset/genderdata/raw/girl2017.txt",
        names=["height(cm)", "weight(kg)", "shoe_size"],
        skiprows=[0],
        sep=r"\s+",
    )
    data2017_Female["sex"] = 0
    data.append(data2017_Female)

    # 2018
    logger.info("Reading 2018 data...")
    data2018 = pd.read_csv(
        "dataset/genderdata/raw/boyandgirl2018.csv",
        names=["height(cm)", "weight(kg)", "shoe_size", "sex"],
        skiprows=[0]
    )
    data.append(data2018)

    # 2021
    logger.info("Reading 2021 data...")
    data2021 = pd.read_csv(
        "dataset/genderdata/raw/2021冬模式识别数据收集.csv",
        names=["index", "sex", "height(cm)", "weight(kg)", "foot_length(mm)", "shoe_size"],
        index_col=0,
    )
    data.append(data2021)

    # 2022
    logger.info("Reading 2022 data...")
    data2022 = pd.read_csv(
        "dataset/genderdata/raw/2022冬模式识别数据收集.csv",
        names=["index", "sex", "height(cm)", "weight(kg)", "foot_length(mm)", "shoe_size"],
        index_col=0,
    )
    data.append(data2022)

    # Merge data
    data = pd.concat(data, ignore_index=True)
    logger.info(f"Data merged. {len(data)} rows in total.")

    # Clean data
    nanData = data[["height(cm)", "weight(kg)", "shoe_size", "sex"]].isnull().any(axis=1)
    data.dropna(subset=["height(cm)", "weight(kg)", "shoe_size", "sex"], inplace=True)
    logger.info(
        f"Dropped {nanData.sum()} rows with NaN values. Left {len(data)} rows."
    )

    data.loc[data["sex"] == "男", "sex"] = 1
    data.loc[data["sex"] == "女", "sex"] = 0
    return data


def preprocess():
    data = read_genderdata()
    savePath = "dataset/genderdata/preprocessed/all.csv"
    if not os.path.exists(os.path.dirname(savePath)):
        os.makedirs(os.path.dirname(savePath))
    data.to_csv(savePath, index=False, encoding="utf-8")
    logger.critical(f"Saved to {savePath}.")


if __name__ == "__main__":
    preprocess()
