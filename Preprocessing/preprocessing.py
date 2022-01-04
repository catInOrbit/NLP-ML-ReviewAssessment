import pandas as pd

def load_dataframe(json_filePath):
    dataframe = pd.read_json(json_filePath, lines=True)
    # dataframe.drop(['verified', 'reviewTime', 'reviewerID', 'asin', 'unixReviewTime',
    #                 'vote', 'image', "style", "reviewerName", "summary"], axis=1, inplace=True)
    # dataframe.dropna(inplace=True)

    dataframe = dataframe.loc[:, ["overall", "reviewText"]]
    dataframe.dropna(inplace=True)
    print(dataframe.describe())
    print(dataframe.info())
    return dataframe

