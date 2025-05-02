import unittest
import os
import shutil
from datetime import datetime
from src.search_manager import SearchManager
from src.vector_store import VectorStore
from src.zhipuai_embedding import ZhipuAIEmbeddings
from langchain_core.documents import Document

class TestSearchManager(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        # 初始化组件，使用内存模式
        self.vector_store = VectorStore(
            persist_directory=None,
            embedding=ZhipuAIEmbeddings()
        )
        self.search_manager = SearchManager(self.vector_store)
        
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
            
    def test_basic_search(self):
        """测试基本搜索功能"""
        # 添加测试文档
        self.vector_store.add_documents(self.test_docs)
        
        # 测试基本搜索
        results = self.search_manager.advanced_search("测试文档", k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("测试文档", results[0].page_content)
        
    def test_score_threshold(self):
        """测试相似度阈值过滤"""
        # 添加测试文档
        self.vector_store.add_documents(self.test_docs)
        
        # 测试相似度阈值过滤（使用更合理的距离阈值）
        results = self.search_manager.advanced_search("测试文档", score_threshold=0.5, k=2)
        self.assertEqual(len(results), 2)
        
    def test_search_history(self):
        """测试搜索历史记录"""
        # 添加测试文档
        self.vector_store.add_documents(self.test_docs)
        
        # 执行多次搜索
        self.search_manager.advanced_search("测试文档")
        self.search_manager.advanced_search("另一个文档")
        
        # 验证搜索历史
        history = self.search_manager.get_search_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["query"], "测试文档")
        self.assertEqual(history[1]["query"], "另一个文档")
        
    def test_similar_queries(self):
        """测试相似查询推荐"""
        # 添加测试文档
        self.vector_store.add_documents(self.test_docs)
        
        # 执行多次搜索
        self.search_manager.advanced_search("测试文档")
        self.search_manager.advanced_search("测试文档")
        self.search_manager.advanced_search("另一个文档")
        
        # 验证相似查询推荐
        similar_queries = self.search_manager.get_similar_queries("测试文档", k=1)
        self.assertEqual(len(similar_queries), 1)
        self.assertEqual(similar_queries[0], "测试文档")

    def test_add_documents(self):
        """测试添加文档"""
        # 添加测试文档
        self.vector_store.add_documents(self.test_docs)
        
        # 测试搜索
        results = self.search_manager.advanced_search("测试文档", k=2)
        self.assertEqual(len(results), 2)

    def test_metadata_filter(self):
        """测试元数据过滤"""
        # 添加测试文档
        self.vector_store.add_documents(self.test_docs)
        
        # 测试元数据过滤
        results = self.search_manager.advanced_search(
            "测试文档",
            filters={"metadata": {"author": "test"}},
            k=2
        )
        self.assertEqual(len(results), 2)

    def test_similarity_search(self):
        """测试相似度搜索"""
        # 添加测试文档
        self.vector_store.add_documents(self.test_docs)
        
        # 测试相似度搜索
        results = self.search_manager.advanced_search("测试文档", k=2)
        self.assertEqual(len(results), 2)

if __name__ == "__main__":
    unittest.main() 