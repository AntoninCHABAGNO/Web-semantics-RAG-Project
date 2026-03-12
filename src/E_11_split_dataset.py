import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "data/kg_clean.txt",
    sep="\t",
    names=["h","r","t"]
)

train, temp = train_test_split(df, test_size=0.2, random_state=42)

# ensure all entities appear in train
entities = set(train["h"]) | set(train["t"])

temp = temp[
    temp["h"].isin(entities) &
    temp["t"].isin(entities)
]

valid, test = train_test_split(temp, test_size=0.5, random_state=42)

train.to_csv("data/train.txt",sep="\t",index=False,header=False)
valid.to_csv("data/valid.txt",sep="\t",index=False,header=False)
test.to_csv("data/test.txt",sep="\t",index=False,header=False)

print("Train:",len(train))
print("Valid:",len(valid))
print("Test:",len(test))