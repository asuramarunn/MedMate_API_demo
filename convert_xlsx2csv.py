import pandas as pd

# Đường dẫn tới file Excel
excel_file = "VOICE/VOICE DATA TRAIN_format.xlsx"

# Đọc tất cả sheet
xls = pd.ExcelFile(excel_file)

# Danh sách để lưu từng DataFrame
all_sheets = []

for sheet_name in xls.sheet_names:
    if sheet_name == "Prompt AI":
        # skip
        continue  
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    all_sheets.append(df)

# Nối tất cả sheet lại
combined_df = pd.concat(all_sheets, ignore_index=True)

# Lưu thành 1 file CSV duy nhất
combined_df.to_csv("audio.csv", index=False)
print("Saved combined.csv")
