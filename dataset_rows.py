
import csv

file_path = "final_data_walmart.csv"

with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)

    total_rows = sum(1 for row in reader)

print(f"Rows + header: {total_rows}")
print(f"Rows - header: {max(0, total_rows - 1)}")