# CSDN详细介绍：https://blog.csdn.net/hahahahahayqq/article/details/141107240?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22141107240%22%2C%22source%22%3A%22hahahahahayqq%22%7D
# 下载原始模型以及int4模型量化
这次看到天池比赛有一些部署上的比赛，练习了一下，主要是实现了int4量化和RAG技术。
ipex_llm这个库好像对Linux的支持要好一点，这里提供官方的环境下载方式，要注意路径的更换

```python
#注意路径，到自己的anaconda的envs
cd  /opt/conda/envs 
mkdir ipex
# 下载 ipex-llm 官方环境|
wget https://s3.idzcn.com/ipex-llm/ipex-llm-2.1.0b20240410.tar.gz 
# 解压文件夹以便恢复原先环境
tar -zxvf ipex-llm-2.1.0b20240410.tar.gz -C ipex/ && rm ipex-llm-2.1.0b20240410.tar.gz
# 安装 ipykernel 并将其注册到 notebook 可使用内核中,注意路径，到自己的anaconda的ipex
/opt/conda/envs/ipex/bin/python3 -m pip install ipykernel && /opt/conda/envs/ipex/bin/python3 -m ipykernel install --name=ipex


###############################还需安装的其他库文件（主要）#####################################
conda activate ipex
pip install sentence_transformers
pip install optimum[openvino,nncf]
pip install langchain
pip install langchain_community
pip install -U huggingface_hub
pip install pypdf
pip install faiss-gpu

```

首先，通过 modelscope 的 api 很容易实现模型的下载


```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
# 第一个参数表示下载模型的型号，第二个参数是下载后存放的缓存地址，第三个表示版本号，默认 master
model_dir = snapshot_download('Qwen/Qwen2-1.5B-Instruct', cache_dir='qwen2chat_src', revision='master')
```

为了实现加速推理，在下载之后需要对Qwen2模型进行精度量化至int4，其实就是对浮点数转换为低位宽的整数型，可以降低对计算资源的需求和提高推理的效率。其中ipex_llm和openvino都对大模型进行了相关优化加速，这次使用openvino、ipex_llm和原本无处理的模型进行一个性能对比还有模型的效果对比，不过量化肯定会造成LLM准确率的下降，不过对于部署边缘设备上这些处理必不可少。

```python
####################ipex_llm##########################
#这段代码的效果是将下载的Qwen2进行sym_int4对称量化，也可以选择非对称量化'asym_int4'
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import  AutoTokenizer
import os
if __name__ == '__main__':
    model_path = os.path.join(os.getcwd(),"qwen2chat_src/Qwen/Qwen2-1___5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int4', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    #将int4模型保存到qwen2chat_int4_ori文件夹里
    model.save_low_bit('qwen2chat_int4_ori')
    tokenizer.save_pretrained('qwen2chat_int4_ori')
```

```python
####################openvino##########################
from optimum.intel import OVModelForCausalLM
from nncf import compress_weights, CompressWeightsMode
model = OVModelForCausalLM.from_pretrained('qwen2chat_src/Qwen/Qwen2-1___5B-Instruct', export=True)
model.model = compress_weights(model.model, mode=CompressWeightsMode.INT4_SYM, group_size=128, ratio=0.8)
model.save_pretrained('qwen2chat_int4')
```
现在，原模型在`qwen2chat_src/Qwen/Qwen2-1___5B-Instruct`，后面的tokenizer会都用原模型包含的分词器，ipex_llm模型保存在`qwen2chat_int4_ori`文件下，openvino模型保存在`qwen2chat_int4`文件夹下。
确保文件正确保存

![](https://i-blog.csdnimg.cn/direct/1bffd07cfbf84223bcd48dbe62ddebde.png#pic_center)

另外为了RAG上还需要用到一个模型实现embedding

```python
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/all-mpnet-base-v2', cache_dir='sentence-transformers')
```


#运行
下载存储库，确保路径没有问题后，运行RAG_trust.py文件
