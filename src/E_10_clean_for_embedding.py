import pandas as pd

df = pd.read_csv(
    "data/kg_triples.txt",
    sep="\t",
    names=["head","relation","tail"]
)

df = df.drop_duplicates()

# remove literals
df = df[df["tail"].str.startswith("http")]

print("Clean triples:",len(df))

df.to_csv("data/kg_clean.txt",sep="\t",index=False,header=False)