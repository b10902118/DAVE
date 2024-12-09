import numpy as np
import matplotlib.pyplot as plt
import os

# 設定目錄
folder_path = "/project/g/r13922043/dave_dataset/FSC147/gt_density_map_adaptive_512_512_object_VarV2"
output_folder = "./visualizations"  # 存放圖片的資料夾

# 建立輸出資料夾（若不存在）
os.makedirs(output_folder, exist_ok=True)

# 獲取所有 npy 檔案
npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

# 遍歷檔案並儲存視覺化圖片
for npy_file in npy_files:
    file_path = os.path.join(folder_path, npy_file)
    data = np.load(file_path)  # 載入數據

    # 視覺化
    plt.figure(figsize=(6, 6))
    plt.title(f"Visualization of {npy_file}")
    plt.imshow(data, cmap="viridis")  # 替換 cmap 調整配色
    plt.colorbar(label="Density")
    plt.axis("off")  # 隱藏坐標

    # 儲存圖片
    output_path = os.path.join(output_folder, f"{npy_file}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # 儲存高解析度圖片
    plt.close()  # 關閉當前圖表以節省記憶體

    print(f"Saved {output_path}")
