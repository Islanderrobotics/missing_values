import pandas as pd
import IslanderDataPreprocessing as IR
from sklearn.impute import SimpleImputer
data  = pd.read_csv("/Users/williammckeon/Sync/youtube videos/dataanalysis/housing.csv")
encoder = IR.Encoder(df = data)
data = encoder.Check()
impute = SimpleImputer(strategy="median")
new_impute = impute.fit_transform(data)
new_data = pd.DataFrame(new_impute, columns= encoder.df.columns, index= encoder.df.index)
IR.DataDiscovery(df = new_data)