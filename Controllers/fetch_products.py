import  os
import pandas as pd
import numpy as np

df = pd.read_excel('Dataset/Product_Final.xlsx')
print (df)

def fetchProducts():
    return df.to_json(orient="records")