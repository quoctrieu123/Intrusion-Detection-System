import pandas as pd
import numpy as np
import ast
import joblib
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from spark_schema import raw_schema, output_schema

# ==========================================
# 1. KHỞI TẠO SPARK SESSION
# ==========================================
spark = SparkSession.builder \
    .appName("CCIOT_Streaming_Preprocessor") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# ==========================================
# 2. ĐỊNH NGHĨA SCHEMA (Mô phỏng)
# ==========================================
# Schema đầu vào (từ Topic raw_flows) - Thay bằng cấu trúc file parquet gốc của bạn
raw_schema = raw_schema  # Đã được định nghĩa trong spark_schema.py, đảm bảo khớp với cấu trúc JSON từ Kafka

# Schema đầu ra (từ hàm mapInPandas ra Topic processed_flows) - Phải khớp 138 cột
output_schema = raw_schema

# ==========================================
# 3. HÀM XỬ LÝ MICRO-BATCH (Pandas UDF)
# ==========================================
def parse_string_to_list(val):
    try:
        if isinstance(val, str):
            return ast.literal_eval(val)
        return val if isinstance(val, list) else []
    except (ValueError, SyntaxError):
        return []
    
def map_to_frequent(proto_list,freq_set):
    res = set()
    for p in proto_list:
        if p in freq_set:
            res.add(p)
        else:
            res.add("other")
    return list(res)

def preprocess_micro_batch(iterator):
    # Khởi tạo artifacts một lần trên mỗi Worker để tối ưu RAM
    scaler = joblib.load(r"C:\Users\Admin\Downloads\IoT Dataset\CCIOT\saved_preprocessed\quantile_scaler.pkl")
    cols_to_drop = joblib.load(r"C:\Users\Admin\Downloads\IoT Dataset\CCIOT\saved_preprocessed\cols_to_drop.pkl")
    numeric_cols = joblib.load(r"C:\Users\Admin\Downloads\IoT Dataset\CCIOT\saved_preprocessed\numeric_columns.pkl")
    # Load các MultiLabelBinarizer đã fit từ lúc train
    mlb_log = joblib.load(r"C:\Users\Admin\Downloads\IoT Dataset\CCIOT\saved_preprocessed\mlb_log_data_types.pkl")
    mlb_src = joblib.load(r"C:\Users\Admin\Downloads\IoT Dataset\CCIOT\saved_preprocessed\mlb_network_protocols_src.pkl")
    mlb_dst = joblib.load(r"C:\Users\Admin\Downloads\IoT Dataset\CCIOT\saved_preprocessed\mlb_network_protocols_dst.pkl")
    freq_src_protos = joblib.load(r"C:\Users\Admin\Downloads\IoT Dataset\CCIOT\saved_preprocessed\freq_network_protocols_src.pkl")
    freq_dst_protos = joblib.load(r"C:\Users\Admin\Downloads\IoT Dataset\CCIOT\saved_preprocessed\freq_network_protocols_dst.pkl")
    
    for pdf in iterator:
        if pdf.empty:
            yield pdf
            continue
        current_drop = [c for c in cols_to_drop if c in pdf.columns]
        pdf.drop(columns=current_drop, inplace=True, errors='ignore')

            
        # B. Parse list và áp dụng One-hot encoding (MultiLabelBinarizer)
        if 'log_data-types' in pdf.columns:
            parsed = pdf['log_data-types'].apply(parse_string_to_list)
            bin_matrix = mlb_log.transform(parsed)
            new_cols = [f"log_type_{c}" for c in mlb_log.classes_]
            df_bin = pd.DataFrame(bin_matrix, columns=new_cols, index=pdf.index)
            pdf = pd.concat([pdf, df_bin], axis=1).drop(columns=['log_data-types'])

        if 'network_protocols_src' in pdf.columns:
            parsed = pdf['network_protocols_src'].apply(parse_string_to_list)
            grouped = parsed.apply(lambda x: map_to_frequent(x, freq_src_protos))
            bin_matrix = mlb_src.transform(grouped)
            new_cols = [f"src_proto_{c}" for c in mlb_src.classes_]
            df_bin = pd.DataFrame(bin_matrix, columns=new_cols, index=pdf.index)
            pdf = pd.concat([pdf, df_bin], axis=1).drop(columns=['network_protocols_src'])

        if 'network_protocols_dst' in pdf.columns:
            parsed = pdf['network_protocols_dst'].apply(parse_string_to_list)
            grouped = parsed.apply(lambda x: map_to_frequent(x, freq_dst_protos))
            bin_matrix = mlb_dst.transform(grouped)
            new_cols = [f"dst_proto_{c}" for c in mlb_dst.classes_]
            df_bin = pd.DataFrame(bin_matrix, columns=new_cols, index=pdf.index)
            pdf = pd.concat([pdf, df_bin], axis=1).drop(columns=['network_protocols_dst'])

        if "timestamp_start" in pdf.columns:
            pdf.rename(columns={"timestamp_start": "timestamp"}, inplace=True)


        # E. Lấy danh sách cột số và áp dụng QuantileTransformer
        numeric_cols = numeric_cols
        if 'label' in numeric_cols: numeric_cols.remove('label')
        if 'timestamp' in numeric_cols: numeric_cols.remove('timestamp')
        
        # Đảm bảo thứ tự cột numeric khớp với lúc fit scaler
        # (Khuyến nghị: Nên lưu danh sách numeric_cols lúc train ra joblib và dùng lại ở đây)
        pdf[numeric_cols] = scaler.transform(pdf[numeric_cols])
        
        expected_cols = output_schema.fieldNames()
        pdf = pdf[expected_cols].fillna(0)
        # Trả về dataframe Pandas cho Spark
        yield pdf

# ==========================================
# 4. KẾT NỐI LUỒNG KAFKA VÀ CHẠY
# ==========================================
# Đọc luồng từ Kafka
df_kafka = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "raw_flows") \
    .load()

# Parse JSON
df_parsed = df_kafka.select(from_json(col("value").cast("string"), raw_schema).alias("data")).select("data.*")

# Áp dụng Pandas UDF
df_processed = df_parsed.mapInPandas(preprocess_micro_batch, schema=output_schema)

# Đóng gói lại thành JSON và đẩy sang topic processed_flows
query = df_processed \
    .selectExpr("to_json(struct(*)) AS value") \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "processed_flows") \
    .option("checkpointLocation", "/tmp/spark_checkpoints_cciot") \
    .start()

query.awaitTermination()