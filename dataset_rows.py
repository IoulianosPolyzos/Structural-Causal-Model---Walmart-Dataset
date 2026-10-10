
import pandas as pd
import time

# ========================
# PARQUET
# ========================
file_path_parquet = "final_data_favorita.parquet"

start = time.perf_counter()
df_parquet = pd.read_parquet(file_path_parquet)
parquet_time = time.perf_counter() - start

print("===== PARQUET =====")
print(f"Rows: {len(df_parquet):,}")
print(f"Columns: {len(df_parquet.columns)}")
print(f"Read time: {parquet_time:.4f} seconds")


# ========================
# CSV
# ========================
file_path_csv = "final_data_favorita.csv"

start = time.perf_counter()
df_csv = pd.read_csv(file_path_csv)
csv_time = time.perf_counter() - start

print("\n===== CSV =====")
print(f"Rows: {len(df_csv):,}")
print(f"Columns: {len(df_csv.columns)}")
print(f"Read time: {csv_time:.4f} seconds")


# ========================
# COMPARISON
# ========================
print("\n===== COMPARISON =====")
print(f"CSV read time: {csv_time:.4f} seconds")
print(f"Parquet read time: {parquet_time:.4f} seconds")

if parquet_time > 0:
    print(f"CSV / Parquet time ratio: {csv_time / parquet_time:.2f}x")

if csv_time > 0:
    print(f"Parquet is {(1 - parquet_time / csv_time) * 100:.2f}% faster than CSV"
          if parquet_time < csv_time
          else f"Parquet is {(parquet_time / csv_time - 1) * 100:.2f}% slower than CSV")
