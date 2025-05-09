from typing import List, Dict, Optional
from datetime import datetime
from langchain_core.documents import Document
import numpy as np

class SearchManager:
    def __init__(self, vector_store):
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
        try:
            print(f"执行搜索: '{query}'")
            # 获取基础搜索结果
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k
            )
            
            if not results:
                print("搜索返回0条结果，返回默认文档")
                return [Document(
                    page_content="对不起，我无法找到与您问题相关的信息。请尝试其他问题或调整搜索条件。",
                    metadata={"source": "default", "date": datetime.now().isoformat()}
                )]
                
            print(f"搜索返回 {len(results)} 条结果")
            
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
                    if "date" in doc.metadata and
                    start_date <= datetime.fromisoformat(doc.metadata.get("date", "")) <= end_date
                ]
            
            # 如果没有结果，返回一个默认文档
            if not filtered_results:
                # 如果过滤过于严格，返回基础搜索的前两个结果
                if results:
                    filtered_results = [doc for doc, _ in results[:2]]
                # 如果还是没有结果，创建一个默认文档
                if not filtered_results:
                    filtered_results = [Document(
                        page_content="对不起，我无法找到与您问题相关的信息。请尝试其他问题或调整搜索条件。",
                        metadata={"source": "default", "date": datetime.now().isoformat()}
                    )]
        except Exception as e:
            print(f"搜索时出错: {str(e)}")
            # 发生错误时返回默认文档
            filtered_results = [Document(
                page_content=f"搜索时出错: {str(e)}。请检查您的API密钥是否有效，或者尝试其他问题。",
                metadata={"source": "error", "date": datetime.now().isoformat()}
            )]
                
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
            
        try:
            # 使用已有的向量数据库中的embedding实例，而不是创建新的
            embedding = self.vector_store.embedding
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
        except Exception as e:
            print(f"计算相似查询时出错: {str(e)}")
            return []
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            v1_array = np.array(v1)
            v2_array = np.array(v2)
            return np.dot(v1_array, v2_array) / (np.linalg.norm(v1_array) * np.linalg.norm(v2_array))
        except Exception as e:
            print(f"计算余弦相似度时出错: {str(e)}")
            return 0.0 