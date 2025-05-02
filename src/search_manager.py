from typing import List, Dict, Optional
from datetime import datetime
from langchain_core.documents import Document
from src.vector_store import VectorStore
from src.zhipuai_embedding import ZhipuAIEmbeddings

class SearchManager:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.search_history = []
        
    def advanced_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        k: int = 4,
        score_threshold: float = 0.5
    ) -> List[Document]:
        """
        高级搜索功能
        
        Args:
            query: 搜索查询
            filters: 过滤条件，例如：
                    {
                        "metadata": {"source": "example.pdf"},
                        "date_range": {
                            "start": "2024-01-01",
                            "end": "2024-12-31"
                        }
                    }
            k: 返回结果数量
            score_threshold: 相似度阈值（距离阈值，越小越好）
            
        Returns:
            List[Document]: 搜索结果列表
        """
        # 获取基础搜索结果
        results = self.vector_store.vectordb.similarity_search_with_score(
            query,
            k=k
        )
        
        # 应用相似度阈值过滤（距离越小越好）
        filtered_results = [
            doc for doc, score in results 
            if score <= score_threshold
        ]
        
        # 应用元数据过滤
        if filters and "metadata" in filters:
            metadata_filters = filters["metadata"]
            filtered_results = [
                doc for doc in filtered_results
                if all(
                    doc.metadata.get(key) == value
                    for key, value in metadata_filters.items()
                )
            ]
            
        # 应用日期范围过滤
        if filters and "date_range" in filters:
            date_range = filters["date_range"]
            start_date = datetime.strptime(date_range["start"], "%Y-%m-%d")
            end_date = datetime.strptime(date_range["end"], "%Y-%m-%d")
            
            filtered_results = [
                doc for doc in filtered_results
                if start_date <= datetime.fromisoformat(doc.metadata.get("date", "")) <= end_date
            ]
            
        # 记录搜索历史
        self.search_history.append({
            "query": query,
            "filters": filters,
            "timestamp": datetime.now(),
            "result_count": len(filtered_results)
        })
        
        return filtered_results
    
    def get_search_history(self, limit: int = 10) -> List[Dict]:
        """获取最近的搜索历史"""
        return self.search_history[-limit:]
    
    def clear_search_history(self):
        """清空搜索历史"""
        self.search_history = []
        
    def get_similar_queries(self, query: str, k: int = 3) -> List[str]:
        """获取相似的历史查询"""
        if not self.search_history:
            return []
            
        # 使用向量相似度计算查询相似度
        embedding = ZhipuAIEmbeddings()
        query_embedding = embedding.embed_query(query)
        
        # 计算历史查询的相似度
        similarities = []
        for history in self.search_history:
            history_embedding = embedding.embed_query(history["query"])
            similarity = self._cosine_similarity(query_embedding, history_embedding)
            similarities.append((history["query"], similarity))
            
        # 返回最相似的k个查询
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [query for query, _ in similarities[:k]]
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度"""
        import numpy as np
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) 