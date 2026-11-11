
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

names = sorted([f for f in os.listdir("data/xrays") if f.endswith(".png")], key=lambda f: int(f[:-4]))
xrays = [f"data/xrays/{f}" for f in names]
masks = [f"data/masks/{f}" if os.path.exists(f"data/masks/{f}") else "None" for f in names]
df = pd.DataFrame({"xrays": xrays, "masks": masks})

train, val = train_test_split(df[df.masks != "None"], test_size=0.2, random_state=42)
csvs = {"dataset": df, "train": train, "val": val, "test": df[df.masks == "None"]}

for name, data in csvs.items():
    data.to_csv(f"data/CSVs/{name}.csv", index=False)

plt.bar(csvs.keys(), [len(v) for v in csvs.values()])
plt.savefig("data/CSVs/csv_counts.png")
plt.show()