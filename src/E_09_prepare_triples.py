from rdflib import Graph

g = Graph()
g.parse("data/expanded_kb.ttl")

triples = []

for s,p,o in g:
    if str(o).startswith("http"):
        triples.append((str(s), str(p), str(o)))

print("Triples:", len(triples))

with open("data/kg_triples.txt","w") as f:
    for h,r,t in triples:
        f.write(f"{h}\t{r}\t{t}\n")