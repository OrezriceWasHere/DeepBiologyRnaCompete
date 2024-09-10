import os, sys
import json

path = "."
dirs = os.listdir( path )

accumulator = {}

for file in filter(lambda x: "RBP" in x ,dirs):
    if "_" in file:
        sequence = file.split("_")[0].replace("RBP","")
        label = file.split("_")[1].replace(".txt","")
        if sequence not in accumulator:
            accumulator[sequence] = {}
        if "train" not in accumulator[sequence]:
            accumulator[sequence]["train"] = []

        accumulator[sequence]["train"].append(file)
    else:
        sequence = file.replace("RBP","").replace(".txt","")
        if sequence not in accumulator:
            accumulator[sequence] = {}
        accumulator[sequence]["test"] ={
            "sequence": file,
            "intensities": "RNAcompete_sequences_rc.txt"
        }

values = list(accumulator.values())
sorted_by_sequence = sorted(values, key=lambda x: int(x["test"]["sequence"].replace("RBP","").replace(".txt","")))

print(json.dumps(list(sorted_by_sequence)))


