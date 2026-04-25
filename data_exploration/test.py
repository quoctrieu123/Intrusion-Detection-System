import pandas as pd
train_df = pd.read_parquet(r"final_data\chunk-based-split-3\train_df_prepared.parquet", engine="pyarrow")

# viết một đoạn code kiểm tra xem liệu dữ liệu có được sắp xếp theo class hay không, nghĩa là các dòng đầu tiên trong train_df thuộc class 1, các dòng tiếp theo thuộc class 2, v.v...

label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]

# kiểm tra dòng đầu tiên mà xuất hiện label = 1, dòng đầu tiên mà xuất hiện label = 2, v.v...
for i in label:
    first_index = train_df[train_df['label'] == i].index[0]
    print(f"Label {i} xuất hiện lần đầu tiên ở dòng index: {first_index}")
    last_index = train_df[train_df['label'] == i].index[-1]
    print(f"Label {i} xuất hiện lần cuối cùng ở dòng index: {last_index}")
