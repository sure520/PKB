import os
from typing import List
import zhipuai
from langchain.embeddings.base import Embeddings

class ZhipuAIEmbeddings(Embeddings):
    def __init__(self):
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("ZHIPUAI_API_KEY not found in environment variables")
        self.client = zhipuai.ZhipuAI(api_key=api_key)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文档转换为向量"""
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model="embedding-2",
                input=text
            )
            if hasattr(response, 'data') and len(response.data) > 0:
                embeddings.append(response.data[0].embedding)
            else:
                raise ValueError(f"Error from ZhipuAI API: {response}")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """将查询转换为向量"""
        response = self.client.embeddings.create(
            model="embedding-2",
            input=text
        )
        if hasattr(response, 'data') and len(response.data) > 0:
            return response.data[0].embedding
        else:
            raise ValueError(f"Error from ZhipuAI API: {response}") 