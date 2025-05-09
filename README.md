# PKB (Personal Knowledge Base)
个人知识库助手

# LangChain: Chat with search

这是一个基于大语言模型的个人知识库助手，可以帮助你管理和查询个人知识库中的内容。本项目使用智谱AI的 embedding-2 模型进行文本向量化，实现高效的知识检索，并使用DeepSeek模型生成回答。

## 功能特点

- 支持多种文档格式（PDF、Markdown、TXT、CSV、JSON）
- 使用智谱AI embedding-2 模型进行文本向量化
- 智能文档分块和向量化存储
- 基于向量数据库的相似度检索
- **全新友好的Web聊天界面，类似ChatGPT风格**
- **无需上传文档也能聊天，有文档时自动使用知识库增强回答**
- 支持中文对话（使用DeepSeek模型）
- 支持在Streamlit Cloud上部署
- **左侧边栏提供API密钥输入和文档上传功能**
- 高级搜索功能
  - 相似度检索
  - 文档片段视图
  - 搜索历史记录

## 在Streamlit Cloud上部署

本项目已经针对Streamlit Cloud进行了优化，可以直接在线部署使用。

### 部署步骤：

1. 在[Streamlit Cloud](https://share.streamlit.io/)上注册并登录
2. 点击"New app"，连接到您的GitHub仓库
3. 选择本项目的仓库和分支
4. 部署完成后，打开应用
5. 在应用左侧面板输入您的智谱AI和DeepSeek API密钥

### 获取API密钥：

- 智谱AI API密钥：在[智谱AI开放平台](https://open.bigmodel.cn/)注册并创建密钥
- DeepSeek API密钥：在[DeepSeek开放平台](https://platform.deepseek.com/)注册并创建密钥

## 本地使用

如果您想在本地环境中使用本项目，请按照以下步骤操作：

### 环境要求

- Python 3.8+
- zhipuai>=2.1.5
- langchain>=0.1.0
- streamlit>=1.24.0
- 其他依赖详见requirements.txt

### 安装说明

1. 克隆项目并进入项目目录：
```bash
git clone https://github.com/your-username/personnal-database.git
cd personnal-database
```

2. 创建并激活虚拟环境：
```bash
# 使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 或使用conda
conda create -n personal_database python=3.8
conda activate personal_database
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 运行应用：
```bash
streamlit run streamlit_app.py
```

## 使用方法

1. 在左侧边栏输入您的API密钥：
   - 智谱AI API密钥（用于文本嵌入，必填）
   - DeepSeek API密钥（用于生成回答，必填）

2. **无需文档也能聊天：**
   - 输入API密钥后即可直接在聊天框中与AI对话
   - 无文档时，使用DeepSeek大模型直接生成回答

3. **上传文档增强回答：**
   - 在左侧边栏上传文档或使用示例数据
   - 支持PDF、Markdown、TXT、CSV、JSON格式
   - 上传文档后自动切换为RAG（检索增强生成）模式
   - 回答将基于知识库内容生成，更加精准

4. 重置功能：
   - 使用"重置聊天历史"清除对话
   - 使用"清除已加载文档"移除知识库

## 注意事项

1. API密钥仅保存在本地会话中，不会被上传到服务器
2. 需要提供DeepSeek API密钥才能启用聊天功能
3. 上传的文档内容仅临时存储在会话中，应用关闭后会被清除
4. 在Streamlit Cloud上部署时，建议添加API密钥到Streamlit Secrets管理中

## 更新日志

### v1.2.0 (2024-06-15)
- 全新UI界面，采用ChatGPT风格的左侧边栏+主聊天区
- 增加无文档聊天功能，不再强制要求加载文档
- 侧边栏集成API密钥输入和文档上传
- 文档状态实时显示
- 优化错误处理和用户体验

### v1.1.0 (2024-06-10)
- 增加Streamlit Cloud部署支持
- 添加用户API密钥输入界面
- 增强错误处理
- 修复与zhipuai 2.1.5的兼容性问题

### v1.0.0 (2024-06-01)
- 初始版本发布
- 支持文档上传和检索
- 支持中文对话
- 支持API密钥配置

## 许可证

MIT License

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
├── data/                  # 存放知识库源文档
├── src/                   # 源代码
│   ├── zhipuai_embedding.py    # 智谱AI Embedding封装
│   ├── document_processor.py   # 文档处理
│   ├── vector_store.py         # 向量数据库管理
│   ├── deepseek_llm.py         # DeepSeek模型封装
│   └── search_manager.py       # 搜索管理
├── vector_db/             # 向量数据库存储目录
├── temp_data/             # 临时文件存储
├── requirements.txt       # 项目依赖
├── streamlit_app.py       # Streamlit应用主程序
├── .env.example           # 环境变量示例配置
└── README.md              # 项目说明
```

## 高级配置

1. 文档分块设置：
   - 在 `document_processor.py` 中修改分块大小和重叠度
   - 默认分块大小为1000字符，重叠度为200字符

2. 检索设置：
   - 在 `search_manager.py` 中修改检索参数
   - 可以调整相似度阈值和返回文档数量

3. 模型参数：
   - 在 `streamlit_app.py` 中修改 DeepSeek 模型参数
   - 可以调整温度（temperature）和最大token数
