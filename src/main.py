import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStore
from zhipuai_embedding import ZhipuAIEmbeddings

def main():
    # 加载环境变量
    load_dotenv()
    
    # 初始化文档处理器
    doc_processor = DocumentProcessor()
    
    # 初始化向量数据库
    embedding = ZhipuAIEmbeddings()
    vector_store = VectorStore(
        embedding=embedding,
        persist_directory="../vector_db"
    )
    
    # 处理文档
    documents = doc_processor.process_documents("../data")
    
    # 创建向量数据库
    vector_store.create_from_documents(documents)
    print(f"向量库中存储的文档数量：{vector_store.get_document_count()}")
    
    # 测试搜索
    query = "什么是机器学习？"
    results = vector_store.similarity_search(query, k=3)
    print(f"\n查询：{query}")
    print("\n搜索结果：")
    for i, doc in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"内容：{doc.page_content[:200]}...")
        print(f"来源：{doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    main() 