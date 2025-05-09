import os
from typing import List
import zhipuai
from langchain.embeddings.base import Embeddings
import numpy as np
import hashlib

class ZhipuAIEmbeddings(Embeddings):
    def __init__(self, api_key=None):
        # 优先使用传入的API密钥，其次从环境变量获取
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            raise ValueError("ZHIPUAI_API_KEY not found in environment variables or parameters")
        
        # 检查是否为测试模式
        self.demo_mode = self.api_key.startswith("test_key_")
        if not self.demo_mode:
            # 适配新版本 API (zhipuai 2.1.5)
            try:
                self.client = zhipuai.ZhipuAI(api_key=self.api_key)
                print(f"ZhipuAI初始化成功，API密钥长度: {len(self.api_key)}")
            except Exception as e:
                print(f"ZhipuAI初始化错误: {str(e)}")
                raise
        else:
            print("使用演示模式，将返回模拟的嵌入向量")
    
    def _get_demo_embedding(self, text: str) -> List[float]:
        """为演示模式生成确定性的模拟嵌入向量"""
        # 使用文本的哈希值生成伪随机但确定性的向量
        hash_object = hashlib.md5(text.encode())
        seed = int(hash_object.hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        # 生成1024维的随机单位向量
        vector = np.random.randn(1024)
        # 归一化为单位向量
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文档转换为向量"""
        # 演示模式
        if self.demo_mode:
            return [self._get_demo_embedding(text) for text in texts]
            
        # 正常API模式
        embeddings = []
        for text in texts:
            try:
                # 适配新版本 API
                response = self.client.embeddings.create(
                    model="embedding-2",
                    input=text
                )
                
                # 检查响应格式并处理
                if hasattr(response, 'data') and len(response.data) > 0:
                    embeddings.append(response.data[0].embedding)
                else:
                    raise ValueError(f"Error from ZhipuAI API: {response}")
            except Exception as e:
                # 提供更详细的错误信息
                print(f"在处理文本嵌入时出错: {str(e)}")
                print(f"问题文本: {text[:100]}...")
                raise
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """将查询转换为向量"""
        # 演示模式
        if self.demo_mode:
            return self._get_demo_embedding(text)
            
        # 正常API模式
        try:
            # 适配新版本 API
            response = self.client.embeddings.create(
                model="embedding-2",
                input=text
            )
            
            # 检查响应格式并处理
            if hasattr(response, 'data') and len(response.data) > 0:
                return response.data[0].embedding
            else:
                raise ValueError(f"Error from ZhipuAI API: {response}")
        except Exception as e:
            # 提供更详细的错误信息
            print(f"在处理查询嵌入时出错: {str(e)}")
            print(f"查询文本: {text}")
            raise 