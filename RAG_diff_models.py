import os
import time
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
# import transformers as trans
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer
from transformers import pipeline
# import torch



def load_model(framework, model_dir, tokenizer_path):
    if framework == 'openvino':

        from optimum.intel import OVModelForCausalLM
        model = OVModelForCausalLM.from_pretrained(model_dir)

    elif framework == 'pytorch':
        from transformers import AutoModelForCausalLM as TAutoModelForCausalLM
        
        model = TAutoModelForCausalLM.from_pretrained(model_dir)

    elif framework == 'intel_llm':

        from ipex_llm.transformers import AutoModelForCausalLM as intelAutoModelForCausalLM
        model = intelAutoModelForCausalLM.load_low_bit(model_dir, trust_remote_code=True)
    
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def run_inference(framework, model_dir, tokenizer_path, pdf_path, query):
    # 记录开始时间
    start_time = time.time()

    # 设置OpenMP线程数为8
    os.environ["OMP_NUM_THREADS"] = "8"

    # 1. 准备模型
    print(f"准备{framework}模型...")
    model_load_start = time.time()
    ragmodel, tokenizer = load_model(framework, model_dir, tokenizer_path)
    model_load_end = time.time()
    print(f"模型加载时间: {model_load_end - model_load_start:.2f} 秒")

    # 2. 创建向量数据库
    print("创建向量数据库...")
    db_creation_start = time.time()
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/AI-ModelScope/all-mpnet-base-v2')
    db = FAISS.from_documents(texts, embeddings)

    # 将创建的向量数据库保存到本地:
    db.save_local("Library2")

    # 加载保存的向量数据库
    loaded_db = FAISS.load_local("Library2", embeddings, allow_dangerous_deserialization=True)
    db_creation_end = time.time()
    print(f"向量数据库创建时间: {db_creation_end - db_creation_start:.2f} 秒")

    # 3. 实现TheRAG系统
    print("实现TheRAG系统...")
    rag_setup_start = time.time()

    
    # 创建pipeline
    pipe = pipeline(
        "text-generation",
        model=ragmodel,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )

    # 创建HuggingFacePipeline实例
    llm = HuggingFacePipeline(pipeline=pipe)

    # 创建RetrievalQA链
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=loaded_db.as_retriever()
    )

    rag_setup_end = time.time()
    print(f"RAG系统设置时间: {rag_setup_end - rag_setup_start:.2f} 秒")

    # 使用示例
    print("执行查询...")
    query_start = time.time()
    print('query:', query)
    # query = "llama2 的实际效果如何"
    result = qa.run(query)
    query_end = time.time()
    
    print(f"查询执行时间: {query_end - query_start:.2f} 秒")

    print("\n查询结果:")
    print(result)

    # 计算总执行时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n总执行时间: {total_time:.2f} 秒")


    result_time = query_end - query_start
    
    return result_time, result

# # 示例调用
# frameworks = ['openvino', 'pytorch', 'intel_llm']
# model_dirs = {
#     'openvino': 'qwen2chat_int4',
#     'pytorch': 'qwen2chat_src/Qwen/Qwen2-1___5B-Instruct',
#     'intel_llm': 'qwen2chat_int4_ori'
# }
# tokenizer_path = "qwen2chat_src/Qwen/Qwen2-1___5B-Instruct"
# pdf_path = "llamatiny.pdf"
# query = "llama2 的实际效果如何"

# for framework in frameworks:
#     print(f"Testing framework: {framework}")
#     model_dir = model_dirs[framework]
#     total_time = run_inference(framework, model_dir, tokenizer_path, pdf_path, query)
#     print(f"Total inference time for {framework}: {total_time:.2f} seconds\n")



# openvino_dir = 'qwen2chat_int4'

# pytorch_dir = 'qwen2chat_src\\Qwen\\Qwen2-1___5B-Instruct'

# intel_llm = 'qwen2chat_int4_ori'

# # mdoel, tokenizer = RAG.load_model('openvino', model_dir=openvino_dir, tokenizer_path=pytorch_dir)

# run_inference('openvino', model_dir=openvino_dir, tokenizer_path=pytorch_dir, pdf_path='llamatiny.pdf', query="llama2 效果如何")

# run_inference()