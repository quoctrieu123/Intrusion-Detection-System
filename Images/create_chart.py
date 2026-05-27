import matplotlib.pyplot as plt
import pandas as pd

# Dữ liệu Use case 3 của bạn
data = {
    'Class': [
        'benign', 'recon', 'dos', 'ddos', 'mitm', 'malware', 'web', 'bruteforce'
    ],
    'Count': [
        136800, 33648, 18420, 18056, 8062, 7541, 2796, 1868
    ]
}

df = pd.DataFrame(data)
df = df.sort_values('Count', ascending=True) # Sắp xếp giảm dần từ trên xuống (cột dài nhất ở trên cùng)

plt.figure(figsize=(10, 6))
# Vẽ biểu đồ ngang với màu đỏ gạch
bars = plt.barh(df['Class'], df['Count'], color='#55A868')

# Kích hoạt thang đo Logarit (Rất quan trọng)
plt.xscale('log') 

# Ghi số lượng cụ thể ở cạnh mỗi cột
for bar in bars:
    width = bar.get_width()
    plt.text(width * 1.1, bar.get_y() + bar.get_height()/2, 
             f'{int(width):,}', va='center', ha='left', fontsize=10)

plt.xlabel('Number of samples (Log Scale)')
plt.title("Use case 3: CIC IIOT 2025: Class Distribution")

plt.xlim(right=df['Count'].max() * 5) # Tránh bị cắt chữ số
plt.tight_layout()
plt.savefig('class_distribution_usecase3.pdf', dpi=300) # Lưu dưới dạng PDF chuẩn bài báo
plt.show()