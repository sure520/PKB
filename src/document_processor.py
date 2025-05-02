import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Any
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    UnstructuredMarkdownLoader,
    TextLoader,
    CSVLoader,
    JSONLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._create_text_splitter()
    
    def _create_text_splitter(self):
        """创建文本分割器"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap
        )
    
    @property
    def chunk_size(self) -> int:
        return self._chunk_size
    
    @chunk_size.setter
    def chunk_size(self, value: int):
        self._chunk_size = value
        self._create_text_splitter()
    
    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap
    
    @chunk_overlap.setter
    def chunk_overlap(self, value: int):
        self._chunk_overlap = value
        self._create_text_splitter()
    
    def load_document(self, file_path: str) -> List[Document]:
        """加载单个文档"""
        file_type = file_path.split('.')[-1].lower()
        try:
            if file_type == 'pdf':
                loader = PyMuPDFLoader(file_path)
            elif file_type == 'md':
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_type == 'txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_type == 'csv':
                loader = CSVLoader(
                    file_path=file_path,
                    encoding='utf-8',
                    csv_args={
                        'delimiter': ',',
                        'quotechar': '"'
                    }
                )
            elif file_type == 'json':
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema='.[]',
                    text_content=False
                )
            else:
                print(f"不支持的文件类型: {file_type}")
                return []
            
            documents = loader.load()
            # 添加日期元数据
            current_date = datetime.now().strftime("%Y-%m-%d")
            for doc in documents:
                if "date" not in doc.metadata:
                    doc.metadata["date"] = current_date
            
            print(f"成功加载文件: {file_path}")
            return documents
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {str(e)}")
            return []
    
    def load_documents(self, folder_path: str) -> List[Document]:
        """加载指定文件夹下的所有文档"""
        file_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        
        documents = []
        for file_path in file_paths:
            documents.extend(self.load_document(file_path))
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档分割成更小的块"""
        return self.text_splitter.split_documents(documents)
    
    def process_documents(self, folder_path: str) -> List[Document]:
        """处理文档的完整流程"""
        documents = self.load_documents(folder_path)
        return self.split_documents(documents) 