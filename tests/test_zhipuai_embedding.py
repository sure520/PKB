import unittest
import os
from src.zhipuai_embedding import ZhipuAIEmbeddings

class TestZhipuAIEmbeddings(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.embedding = ZhipuAIEmbeddings()
    
    def test_embed_documents(self):
        """测试文档嵌入功能"""
        # 测试文档列表
        documents = ["这是一个测试文档", "这是另一个测试文档"]
        
        # 获取嵌入
        embeddings = self.embedding.embed_documents(documents)
        
        # 验证结果
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 1024)  # 假设嵌入维度为1024
        self.assertIsInstance(embeddings[0], list)
        self.assertTrue(all(isinstance(x, float) for x in embeddings[0]))
        
    def test_embed_query(self):
        """测试查询嵌入功能"""
        # 测试查询
        query = "测试查询"
        
        # 获取嵌入
        embedding = self.embedding.embed_query(query)
        
        # 验证结果
        self.assertEqual(len(embedding), 1024)  # 假设嵌入维度为1024
        self.assertIsInstance(embedding, list)
        self.assertTrue(all(isinstance(x, float) for x in embedding))
        
    def test_similarity(self):
        """测试相似度计算"""
        # 测试文档
        doc1 = "这是一个测试文档"
        doc2 = "这是另一个测试文档"
        
        # 获取嵌入
        emb1 = self.embedding.embed_query(doc1)
        emb2 = self.embedding.embed_query(doc2)
        
        # 计算余弦相似度
        similarity = sum(a * b for a, b in zip(emb1, emb2)) / (
            (sum(a * a for a in emb1) ** 0.5) * (sum(b * b for b in emb2) ** 0.5)
        )
        
        # 相似度应该在 [-1, 1] 范围内
        self.assertGreaterEqual(similarity, -1)
        self.assertLessEqual(similarity, 1)

if __name__ == "__main__":
    unittest.main() 