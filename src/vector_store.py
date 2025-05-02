from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import os

class VectorStore:
    def __init__(self, persist_directory: Optional[str], embedding: Embeddings):
        self.embedding = embedding
        self.persist_directory = persist_directory
        
        if persist_directory:
            # 确保目录存在并设置权限
            os.makedirs(self.persist_directory, exist_ok=True)
            os.chmod(self.persist_directory, 0o777)
            
            # 初始化向量数据库
            self.vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding
            )
        else:
            # 使用内存模式
            self.vectordb = Chroma(
                embedding_function=self.embedding
            )
    
    def create_from_documents(self, documents: List[Document]):
        """从文档创建向量数据库"""
        if self.persist_directory:
            self.vectordb = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                persist_directory=self.persist_directory
            )
            self.vectordb.persist()
        else:
            self.vectordb = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding
            )
    
    def add_documents(self, documents: List[Document]):
        """添加文档到向量数据库"""
        if not self.vectordb:
            self.create_from_documents(documents)
        else:
            self.vectordb.add_documents(documents)
            if self.persist_directory:
                self.vectordb.persist()
    
    def load_existing(self):
        """加载已存在的向量数据库"""
        if not self.persist_directory:
            raise ValueError("Cannot load database in memory mode")
        self.vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding
        )
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """相似度搜索"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")
        if filter:
            return self.vectordb.similarity_search(query, k=k, filter=filter)
        return self.vectordb.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """带分数的相似度搜索"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")
        return self.vectordb.similarity_search_with_score(query, k=k)
    
    def get_document_count(self) -> int:
        """获取文档数量"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")
        return self.vectordb._collection.count()
        
    def persist(self):
        """持久化向量数据库"""
        if self.vectordb and self.persist_directory:
            self.vectordb.persist() 