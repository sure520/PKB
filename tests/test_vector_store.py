import unittest
import os
import shutil
from datetime import datetime
from langchain_core.documents import Document
from src.vector_store import VectorStore
from src.zhipuai_embedding import ZhipuAIEmbeddings

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        # 使用内存模式
        self.vector_store = VectorStore(
            persist_directory=None,
            embedding=ZhipuAIEmbeddings()
        )
        
        # 创建测试文档
        self.test_docs = [
            Document(
                page_content="这是一个测试文档1",
                metadata={
                    "source": "test1.txt",
                    "author": "test",
                    "date": datetime.now().isoformat()
                }
            ),
            Document(
                page_content="这是另一个测试文档2",
                metadata={
                    "source": "test2.txt",
                    "author": "test",
                    "date": datetime.now().isoformat()
                }
            )
        ]
        
    def test_add_documents(self):
        """测试添加文档功能"""
        # 添加测试文档
        self.vector_store.add_documents(self.test_docs)
        
        # 验证文档是否被正确添加
        results = self.vector_store.similarity_search("测试文档", k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("测试文档", results[0].page_content)
        
    def test_similarity_search(self):
        """测试相似度搜索功能"""
        # 添加测试文档
        self.vector_store.add_documents(self.test_docs)
        
        # 测试相似度搜索
        results = self.vector_store.similarity_search("测试文档", k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("测试文档", results[0].page_content)
        
    def test_persist_and_load(self):
        """测试持久化和加载功能"""
        # 创建临时目录
        test_dir = "test_persist"
        os.makedirs(test_dir, exist_ok=True)
        
        try:
            # 创建持久化存储
            persist_store = VectorStore(
                persist_directory=test_dir,
                embedding=ZhipuAIEmbeddings()
            )
            
            # 添加文档并持久化
            persist_store.add_documents(self.test_docs)
            persist_store.persist()
            
            # 创建新的向量存储实例
            new_store = VectorStore(
                persist_directory=test_dir,
                embedding=ZhipuAIEmbeddings()
            )
            
            # 验证文档是否被正确加载
            results = new_store.similarity_search("测试文档", k=2)
            self.assertEqual(len(results), 2)
            self.assertIn("测试文档", results[0].page_content)
        finally:
            # 清理临时目录
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
        
    def test_metadata_filter(self):
        """测试元数据过滤功能"""
        # 添加测试文档
        self.vector_store.add_documents(self.test_docs)
        
        # 测试元数据过滤
        results = self.vector_store.similarity_search(
            "测试文档",
            filter={"author": "test"},
            k=2
        )
        self.assertEqual(len(results), 2)
        self.assertIn("测试文档", results[0].page_content)
        
if __name__ == "__main__":
    unittest.main() 