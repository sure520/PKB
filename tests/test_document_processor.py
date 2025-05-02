import unittest
import os
import shutil
from datetime import datetime
from langchain_core.documents import Document
from src.document_processor import DocumentProcessor
from src.zhipuai_embedding import ZhipuAIEmbeddings

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试目录
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_data")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 创建测试文件
        self.test_file = os.path.join(self.test_dir, "test.txt")
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("这是一个测试文档。\n这是第二行。\n这是第三行。")
            
        # 初始化文档处理器
        self.processor = DocumentProcessor(
            chunk_size=100,
            chunk_overlap=20
        )
    
    def tearDown(self):
        """测试后的清理工作"""
        # 删除测试目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_load_document(self):
        """测试文档加载功能"""
        # 加载文档
        docs = self.processor.load_document(self.test_file)
        self.assertEqual(len(docs), 1)
        self.assertIn("测试文档", docs[0].page_content)
        
    def test_chunk_size(self):
        """测试文档分块大小"""
        # 修改分块大小
        self.processor.chunk_size = 50
        
        # 加载并分块文档
        docs = self.processor.load_document(self.test_file)
        chunks = self.processor.split_documents(docs)
        
        # 验证分块大小
        self.assertTrue(all(len(chunk.page_content) <= 50 for chunk in chunks))
        
    def test_chunk_overlap(self):
        """测试文档分块重叠"""
        # 修改分块大小和重叠
        self.processor.chunk_size = 50
        self.processor.chunk_overlap = 20
        
        # 加载并分块文档
        docs = self.processor.load_document(self.test_file)
        chunks = self.processor.split_documents(docs)
        
        # 验证分块重叠
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i].page_content
            next_chunk = chunks[i + 1].page_content
            overlap = len(set(current_chunk.split()) & set(next_chunk.split()))
            self.assertGreaterEqual(overlap, 5)  # 至少应该有5个重叠的词

if __name__ == "__main__":
    unittest.main() 