import RAG_diff_models as RAG
import pandas as pd
import os
from datetime import datetime

openvino_dir = 'qwen2chat_int4'

pytorch_dir = 'qwen2chat_src/Qwen/Qwen2-1___5B-Instruct'

intel_llm = 'qwen2chat_int4_ori'

query = "llama2 的实际效果如何"
# mdoel, tokenizer = RAG.load_model('openvino', model_dir=openvino_dir, tokenizer_path=pytorch_dir)
# time = RAG.run_inference('openvino', model_dir=openvino_dir, tokenizer_path=pytorch_dir, pdf_path='llamatiny.pdf', query=query)
openvino_total_time, openvino_result = RAG.run_inference('openvino', model_dir=openvino_dir, tokenizer_path=pytorch_dir, pdf_path='llamatiny.pdf', query = query)
pytorch_total_time, pytorch_result = RAG.run_inference('pytorch', model_dir=pytorch_dir, tokenizer_path=pytorch_dir, pdf_path='llamatiny.pdf', query = query)
intel_llm_total_time, intel_llm_result = RAG.run_inference('intel_llm', model_dir=intel_llm, tokenizer_path=intel_llm, pdf_path='llamatiny.pdf', query = query)
# openvino_total_time = 1
# pytorch_total_time = 1
# openvino_result = 1
# pytorch_result = 1
# 创建一个数据字典
data = {
    "模型": ["OpenVINO", "PyTorch", "Intel LLM"],
    "总耗时 (秒)": [openvino_total_time, pytorch_total_time, intel_llm_total_time],
    "生成结果": [openvino_result, pytorch_result, intel_llm_result]
}

# 使用 pandas 创建一个 DataFrame
df = pd.DataFrame(data)

# 打印 DataFrame 以表格形式展示对比结果
print(df)

# 找出最快的模型
fastest_model = df.loc[df["总耗时 (秒)"].idxmin()]["模型"]
print(f"\n最快的模型是: {fastest_model}")

# 创建一个保存结果的目录
results_dir = "model_comparison_results"
os.makedirs(results_dir, exist_ok=True)

# 生成带有时间戳的文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"model_comparison_{timestamp}.csv"
csv_path = os.path.join(results_dir, csv_filename)

# 保存 DataFrame 到 CSV 文件
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到: {csv_path}")

# 将最快模型信息添加到 CSV 文件
with open(csv_path, 'a', encoding='utf-8-sig') as f:
    f.write(f"\n最快的模型是:,{fastest_model}")

print("CSV 文件已更新，包含最快模型信息。")

