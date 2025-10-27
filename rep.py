import json, os
import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "./data")
FILE_PATH = os.path.join(DATA_DIR, "sim_closed.json")

with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# sadece geçerli kolonlar
cols = ["symbol","dir","power","exit_reason","gain_pct","duration_sec"]
df = df[[c for c in cols if c in df.columns]]

# Power aralıklarını sınıflandıralım
bins = [0,60,70,80,90,100]
labels = ["<60","60-70","70-80","80-90",">90"]
df["power_band"] = pd.cut(df["power"], bins=bins, labels=labels, include_lowest=True)

# TP/SL sayılarını ve ortalama süreleri çıkar
summary = df.groupby(["power_band","exit_reason"]).agg(
    trade_count=("exit_reason","count"),
    avg_gain_pct=("gain_pct","mean"),
    avg_duration_min=("duration_sec", lambda x: (x.mean()/60) if len(x)>0 else 0)
).reset_index()

# TP oranı
pivot = df.pivot_table(index="power_band", columns="exit_reason", values="gain_pct", aggfunc="count", fill_value=0)
pivot["TP_Rate(%)"] = (pivot["TP"] / (pivot["TP"] + pivot["SL"] + 1e-6)) * 100
pivot = pivot.reset_index()

print("==== POWER BANDS PERFORMANCE ====")
print(summary)
print("\n==== TP RATE BY POWER ====")
print(pivot)
