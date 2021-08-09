"""
Preprocess csv file as this format:
https://cloud.google.com/vision/automl/object-detection/docs/csv-format
"""
# %%

from numpy import random
import pandas as pd
import numpy as np
import collections

CSV_PATH = "../images/_annotations.csv"
NEW_CSV_PATH = "../images/_annotations_new.csv"

df = pd.read_csv(CSV_PATH)
headers = ["set", "path", "label", "x_min", "y_min", "", "", "x_max", "y_max", "", ""]
data = []
# %%
random_dict = collections.defaultdict(str)
for id, row in df.iterrows():
    fn = row["filename"]
    if fn not in random_dict:
        r = np.random.random()
        if r < 0.75:
            random_dict[fn] = "TRAIN"
        else:
            r = np.random.random()
            if r < 0.6:
                random_dict[fn] = "VALIDATION"
            else:
                random_dict[fn] = "TEST"
# %%

for id, row in df.iterrows():
    height = row["height"]
    width = row["width"]
    datum = [
        random_dict[row["filename"]],
        row["filename"],
        row["class"],
        row["xmin"] / width,
        row["ymin"] / height,
        "",
        "",
        row["xmax"] / width,
        row["ymax"] / height,
        "",
        "",
    ]

    data.append(datum)

# %%

new_df = pd.DataFrame(data)

new_df.to_csv(NEW_CSV_PATH, header=False, index=False)
