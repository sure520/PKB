# PKB
个人知识库助手
# personnal-database
# 个人知识库助手

这是一个基于大语言模型的个人知识库助手，可以帮助你管理和查询个人知识库中的内容。本项目使用智谱AI的 embedding-2 模型进行文本向量化，实现高效的知识检索。

## 功能特点

- 支持多种文档格式（PDF、Word、TXT、Markdown等）
- 使用智谱AI embedding-2 模型进行文本向量化
- 智能文档分块和向量化存储
- 基于向量数据库的相似度检索
- 友好的Web聊天界面
- 支持中文对话（使用DeepSeek模型）
- 高级搜索功能
  - 相似度阈值调节
  - 日期范围过滤
  - 元数据过滤
  - 搜索历史记录
  - 相似查询推荐

## 智谱AI Embedding 说明

本项目使用智谱AI的 embedding-2 模型进行文本向量化，具有以下特点：

- 支持批量文档向量化
- 支持单条查询向量化
- 向量维度：1024
- 支持中英文等多语言
- 适合中文场景的语义理解

使用示例：
```python
from src.zhipuai_embedding import ZhipuAIEmbeddings

# 初始化 embedding 模型
embeddings = ZhipuAIEmbeddings()

# 文档向量化
doc_vectors = embeddings.embed_documents(["文档内容1", "文档内容2"])

# 查询向量化
query_vector = embeddings.embed_query("查询内容")
```

## 项目结构

```
personal_knowledge_base/
├── data/                # 存放知识库源文档
├── src/                # 源代码
│   ├── zhipuai_embedding.py  # 智谱AI Embedding封装
│   ├── document_processor.py  # 文档处理
│   ├── vector_store.py       # 向量数据库管理
│   ├── deepseek_llm.py      # DeepSeek模型封装
│   ├── search_manager.py    # 搜索管理
│   ├── chat_app.py         # 聊天界面
│   └── main.py             # 主程序
├── vector_db/            # 向量数据库存储目录
├── tests/               # 测试用例
├── requirements.txt     # 项目依赖
├── setup_env.sh        # 环境配置脚本
├── .env                # 环境变量配置
└── README.md           # 项目说明
```

## 环境要求

- Python 3.8+
- zhipuai>=1.0.7
- langchain>=0.1.0
- streamlit>=1.24.0
- python-dotenv

## 安装说明

1. 克隆项目并进入项目目录：
```bash
git clone https://github.com/sure520/personnal-database.git
cd personnal-database
```

2. 创建并激活 conda 环境（推荐）：
```bash
conda create -n personal_database python=3.8
conda activate personal_database
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置环境变量：
   - 在项目根目录创建 `.env` 文件
   - 添加以下配置：
     ```
     ZHIPUAI_API_KEY=你的智谱AI_API密钥
     DEEPSEEK_API_KEY=你的DeepSeek_API密钥
     ```

5. 运行环境配置脚本：
```bash
source setup_env.sh
```

## 使用方法

1. 构建知识库：
```bash
python src/main.py
```

2. 启动聊天界面：
```bash
streamlit run src/chat_app.py
```

3. 在浏览器中打开显示的地址（默认为 http://localhost:8501）

4. 使用搜索功能：
   - 在侧边栏调整相似度阈值（0.0-1.0）
   - 设置日期范围过滤
   - 查看搜索历史记录
   - 获取相似查询推荐

## 高级配置

1. 文档分块设置：
   - 在 `document_processor.py` 中修改分块大小和重叠度
   - 默认分块大小为1000字符，重叠度为200字符

2. 检索设置：
   - 在 `search_manager.py` 中修改检索参数
   - 可以调整相似度阈值和返回文档数量

3. 模型参数：
   - 在 `chat_app.py` 中修改 DeepSeek 模型参数
   - 可以调整温度（temperature）和最大token数

## 注意事项

1. 首次运行需要构建知识库，这可能需要一些时间
2. 确保有足够的磁盘空间存储向量数据库
3. 注意API使用限制和费用
4. 定期备份知识库文档和向量数据库

## 许可证

MIT License
