import pandas as pd
import numpy as np

print("Loading datasets...")
data_test = pd.read_parquet(r"final_data\test_df_prepared.parquet")
data_train = pd.read_parquet(r"final_data\train_df_prepared.parquet")
data_valid = pd.read_parquet(r"final_data\valid_df_prepared.parquet")

print("\n--- Label Distribution ---")
print("Train:")
print(data_train["label"].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
print("\nValid:")
print(data_valid["label"].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
print("\nTest:")
print(data_test["label"].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

print("\n--- Analyzing Feature Distribution Drift (Train vs Test) ---")
numeric_cols = data_train.select_dtypes(include=[np.number]).columns.tolist()
if 'label' in numeric_cols:
    numeric_cols.remove('label')

drift_stats = []
for col in numeric_cols:
    train_mean, train_std = data_train[col].mean(), data_train[col].std()
    test_mean, test_std = data_test[col].mean(), data_test[col].std()
    valid_mean, valid_std = data_valid[col].mean(), data_valid[col].std()
    
    # Calculate percentage difference in mean
    mean_diff_pct = abs(train_mean - test_mean) / (abs(train_mean) + 1e-9) * 100
    
    # Calculate difference in Std
    std_diff_pct = abs(train_std - test_std) / (abs(train_std) + 1e-9) * 100
    
    drift_stats.append({
        'Feature': col,
        'Train_Mean': train_mean,
        'Test_Mean': test_mean,
        'Mean_Diff_%': mean_diff_pct,
        'Train_Std': train_std,
        'Test_Std': test_std,
        'Std_Diff_%': std_diff_pct
    })

drift_df = pd.DataFrame(drift_stats)
drift_df = drift_df.sort_values(by='Mean_Diff_%', ascending=False)

print(f"\nTop 20 Features with the Highest Mean Difference (%) between Train and Test:")
print(drift_df[['Feature', 'Train_Mean', 'Test_Mean', 'Mean_Diff_%', 'Std_Diff_%']].head(20).to_string(index=False))

severe_mean_drift = len(drift_df[drift_df['Mean_Diff_%'] > 20])
severe_std_drift = len(drift_df[drift_df['Std_Diff_%'] > 20])

print(f"\nSummary:")
print(f"Total numerical features analyzed: {len(numeric_cols)}")
print(f"Features with > 20% shift in Mean: {severe_mean_drift}")
print(f"Features with > 20% shift in Std Deviation: {severe_std_drift}")
if severe_mean_drift > len(numeric_cols) * 0.1:
    print("\nWARNING: Significant distribution drift detected! The time-based split likely caused the test set to have fundamentally different traffic patterns/volumes compared to the train set.")