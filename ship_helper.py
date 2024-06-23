import pandas as pd, numpy as np

def extractTitle(name):
    if "." in name:
        return(name.split(".")[0].split(",")[1].strip())
    else:
        return("No title")

def cleanShipData(df):
    df['Title'] = df['Name'].apply(extractTitle)

    # fill the missing age value
    df["Age"] = df.groupby('Title')['Age'].transform(lambda x:x.fillna(x.median()))

    # dropping Cabin as it has too many missing values
    df.drop(columns='Cabin',inplace=True)

    # deleting rows where embarked is missing
    df.dropna(subset=['Embarked'],inplace=True)

    # create TravelAlone
    df['Travelalone'] =  np.where((df['SibSp'] + df['Parch']) > 0, 0 , 1).astype('uint8')

    # dropping features which are not relevant
    df.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Title'],inplace=True)

    df_dummy = pd.get_dummies(df,columns=['Sex','Embarked'],drop_first=True,dtype = int)
    return(df_dummy)