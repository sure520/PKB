import streamlit as st
import os
import sys
from dotenv import load_dotenv
import uuid

print("当前Python解释器路径:", sys.executable)

# 设置页面配置 - 必须是第一个Streamlit命令
st.set_page_config(
    page_title="LangChain: Chat with search",
    page_icon="🦜",
    layout="wide"
)

# 添加 src 目录到 Python 路径
sys.path.append("./src")

# 初始化会话状态 - 需要更全面的初始化
def initialize_session_state():
    """初始化所有会话状态变量"""
    session_state_keys = {
        "chat_messages": [],
        "documents_loaded": False,
        "app_initialized": False,
        "initialization_error": None,
        "show_zhipu_key": False,
        "zhipu_api_key": "",
        "show_key": False,
        "deepseek_api_key": "",
        "documents": [],
        "embedding": None,
        "doc_processor": None,
        "vector_store": None,
        "search_manager": None,
        "llm": None,
        "condense_question_prompt": None,
        "qa_prompt": None
    }
    
    for key, default_value in session_state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# 在页面配置之后立即初始化会话状态
initialize_session_state()

# 侧边栏：API Key输入、文档上传、重置聊天历史
st.sidebar.title("设置")

# 智谱API Key输入（支持明文/密文切换）
zhipu_api_key = st.sidebar.text_input(
    "智谱AI API Key",
    value=st.session_state.get("zhipu_api_key", ""),  # 使用 get() 方法
    type="password" if not st.session_state.get("show_zhipu_key", False) else "default",
    placeholder="请输入智谱API Key"
)
show_zhipu_key = st.sidebar.checkbox("显示智谱API Key", value=st.session_state.get("show_zhipu_key", False))
st.session_state["show_zhipu_key"] = show_zhipu_key
st.session_state["zhipu_api_key"] = zhipu_api_key
os.environ["ZHIPUAI_API_KEY"] = zhipu_api_key if zhipu_api_key else ""

# DeepSeek API Key输入（支持明文/密文切换）
deepseek_api_key = st.sidebar.text_input(
    "DeepSeek API Key",
    value=st.session_state.get("deepseek_api_key", ""),  # 使用 get() 方法
    type="password" if not st.session_state.get("show_key", False) else "default",
    placeholder="请输入DeepSeek API Key"
)
show_key = st.sidebar.checkbox("显示DeepSeek API Key", value=st.session_state.get("show_key", False))
st.session_state["show_key"] = show_key
st.session_state["deepseek_api_key"] = deepseek_api_key
os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key if deepseek_api_key else ""

# 文档上传（支持多种格式）
st.sidebar.subheader("文档上传")
uploaded_files = st.sidebar.file_uploader(
    "上传文档（支持pdf, md, txt, csv, json）",
    type=["pdf", "md", "txt", "csv", "json"],
    accept_multiple_files=True
)

# 处理文档上传
if uploaded_files:
    if not st.session_state.documents_loaded:
        with st.sidebar:
            with st.spinner("处理上传文件..."):
                try:
                    # 确保应用初始化完成
                    if not st.session_state.app_initialized:
                        st.error("请先输入API密钥并等待应用初始化完成")
                    else:
                        # 处理所有上传的文件
                        all_documents = []
                        for uploaded_file in uploaded_files:
                            # 为上传文件生成一个安全的ASCII文件名，保留原始扩展名
                            file_extension = os.path.splitext(uploaded_file.name)[1]
                            safe_filename = f"file_{uuid.uuid4().hex}{file_extension}"
                            # 保存上传的文件（只用safe_filename，避免任何非ASCII字符）
                            temp_file_path = os.path.join(os.getcwd(), "temp_data", safe_filename)
                            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
                            with open(temp_file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            # 处理文件
                            documents = st.session_state.doc_processor.load_document(temp_file_path)
                            documents = st.session_state.doc_processor.split_documents(documents)
                            # 为每个文档片段添加原始文件名到元数据
                            for doc in documents:
                                if not hasattr(doc, 'metadata'):
                                    doc.metadata = {}
                                doc.metadata['source_file'] = uploaded_file.name
                                doc.metadata['file_size'] = len(uploaded_file.getbuffer())
                                doc.metadata['file_type'] = file_extension.lstrip('.')
                                doc.metadata['saved_filename'] = safe_filename
                            all_documents.extend(documents)
                    
                        if all_documents:
                            # 添加到向量库
                            st.session_state.vector_store.create_from_documents(all_documents)
                            st.session_state.documents = all_documents
                            st.session_state.documents_loaded = True
                            st.success(f"✅ 成功处理文件，加载了 {len(all_documents)} 个文档片段")
                        else:
                            st.warning("⚠️ 未能从文件中提取文档")
                except Exception as e:
                    st.error(f"❌ 处理上传文件时出错: {str(e)}")
    else:
        st.sidebar.success(f"✅ 已加载 {len(st.session_state.documents)} 个文档片段")

# 示例数据按钮
st.sidebar.subheader("示例数据")
if st.sidebar.button("加载示例数据"):
    with st.sidebar.spinner("加载示例数据..."):
        try:
            # 确保应用初始化完成
            if not st.session_state.app_initialized:
                st.sidebar.error("请先输入API密钥并等待应用初始化完成")
            else:
                example_text = """
                # 机器学习基础
                
                机器学习是人工智能的一个分支，它使用算法和统计模型让计算机系统能够执行特定任务，而无需使用明确的指令。
                
                ## 主要类型
                
                1. 监督学习：使用标记的训练数据进行学习
                2. 无监督学习：在没有标记的数据中找出结构
                3. 强化学习：通过与环境交互来学习最佳行动
                
                ## 常见算法
                
                机器学习算法包括线性回归、决策树、随机森林、支持向量机和神经网络。
                """
                
                # 创建临时文件
                temp_file_path = os.path.join(os.getcwd(), "temp_data", "example.md")
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
                
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    f.write(example_text)
                
                # 处理文件
                documents = st.session_state.doc_processor.load_document(temp_file_path)
                documents = st.session_state.doc_processor.split_documents(documents)
                
                if documents:
                    # 添加到向量库
                    st.session_state.vector_store.create_from_documents(documents)
                    st.session_state.documents = documents
                    st.session_state.documents_loaded = True
                    st.sidebar.success(f"✅ 成功加载示例数据，得到 {len(documents)} 个文档片段")
                else:
                    st.sidebar.warning("⚠️ 未能从示例数据中提取文档")
        except Exception as e:
            st.sidebar.error(f"❌ 处理示例数据时出错: {str(e)}")

# 重置聊天历史按钮
st.sidebar.subheader("聊天设置")
if st.sidebar.button("重置聊天历史"):
    st.session_state["chat_messages"] = []
    st.sidebar.success("聊天历史已重置")

# 重置文档按钮
if st.session_state.documents_loaded:
    if st.sidebar.button("清除已加载文档"):
        st.session_state.documents_loaded = False
        if "documents" in st.session_state:
            del st.session_state.documents
        if "vector_store" in st.session_state and hasattr(st.session_state.vector_store, "clear"):
            st.session_state.vector_store.clear()
        st.sidebar.success("✅ 已清除所有文档")
        st.experimental_rerun()

# 主区：Logo+标题
st.markdown("""
<div style='text-align: center;'>
    <span style='font-size:48px;'>🦜</span>
    <h1>LangChain: Chat with search</h1>
</div>
""", unsafe_allow_html=True)

# 显示环境信息
with st.expander("系统信息", expanded=False):
    st.write(f"Python 版本: {sys.version}")
    st.write(f"运行路径: {os.getcwd()}")
    
    # 尝试手动加载 .env 文件
    try:
        load_dotenv(override=True)  # 使用 override=True 确保覆盖已存在的环境变量
        st.success("✅ 已加载 .env 文件")
    except Exception as e:
        st.error(f"❌ 加载 .env 文件出错: {str(e)}")

# 显示API密钥状态
api_status_col1, api_status_col2 = st.columns(2)
with api_status_col1:
    if st.session_state.get("zhipu_api_key"):
        st.success("✅ 智谱AI API密钥已设置")
    else:
        st.error("❌ 智谱AI API密钥未设置")
        
with api_status_col2:
    if st.session_state.get("deepseek_api_key"):
        st.success("✅ DeepSeek API密钥已设置")
    else:
        st.error("❌ DeepSeek API密钥未设置")

# 主区逻辑：未输入DeepSeek API Key时禁用聊天输入并提示
if not deepseek_api_key:
    st.info("请在左侧输入 DeepSeek API Key 以启用聊天功能。")
    st.chat_input("请输入您的问题", disabled=True)
    st.stop()  # 停止执行后续代码

# 应用主体部分
try:
    # 初始化组件
    if not st.session_state.app_initialized:
        with st.spinner("正在初始化应用组件..."):
            try:
                # 初始化 embedding
                from src.zhipuai_embedding import ZhipuAIEmbeddings
                embedding = ZhipuAIEmbeddings(api_key=st.session_state.get("zhipu_api_key", ""))
                st.session_state.embedding = embedding
                
                # 初始化文档处理器
                from src.document_processor import DocumentProcessor
                doc_processor = DocumentProcessor()
                st.session_state.doc_processor = doc_processor
                
                # 初始化向量存储
                from src.vector_store import VectorStore
                
                # 创建临时目录用于测试
                temp_dir = os.path.join(os.getcwd(), "temp_vector_db")
                os.makedirs(temp_dir, exist_ok=True)
                
                vector_store = VectorStore(
                    embedding=st.session_state.embedding,
                    persist_directory=temp_dir
                )
                st.session_state.vector_store = vector_store
                
                # 初始化搜索管理器
                from src.search_manager import SearchManager
                search_manager = SearchManager(st.session_state.vector_store)
                st.session_state.search_manager = search_manager
                
                # 初始化语言模型
                from src.deepseek_llm import DeepSeekChat
                use_demo_mode = not st.session_state.get("deepseek_api_key", "")
                llm = DeepSeekChat(
                    model="deepseek-chat",
                    temperature=0.7,
                    max_tokens=2000,
                    api_key=st.session_state.get("deepseek_api_key", ""),
                    demo_mode=use_demo_mode  # 如果没有API密钥就自动使用演示模式
                )
                st.session_state.llm = llm
                
                # 初始化聊天模板
                from langchain_core.prompts import ChatPromptTemplate
                
                # 问题总结模板
                condense_question_system_template = (
                    "请根据聊天记录总结用户最近的问题，"
                    "如果没有多余的聊天记录则返回用户的问题。"
                    "\n\n聊天记录: {chat_history}"
                )
                condense_question_prompt = ChatPromptTemplate.from_messages([
                    ("system", condense_question_system_template),
                    ("human", "{input}"),
                ])
                st.session_state.condense_question_prompt = condense_question_prompt
                
                # 回答生成模板
                system_prompt = (
                    "你是一个基于知识库的问答助手。 "
                    "请使用检索到的上下文片段回答这个问题，如果用户的输入包含‘全文’等有关内容，\
                        那么你需要结合具体文档的全文内容进行回答。 "
                    "如果你不知道答案就说不知道。 "
                    "请使用简洁的话语回答用户。"
                    "在回答时，如果引用了特定文档，请说明文档来源。"
                    "\n\n上下文: {context}"
                    "\n\n聊天记录: {chat_history}"
                )
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                st.session_state.qa_prompt = qa_prompt
                
                st.session_state.app_initialized = True
                st.success("🎉 应用初始化完成！")
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.session_state.initialization_error = str(e)
                st.error(f"❌ 初始化应用时出错: {str(e)}")
                st.code(error_details, language="python")
                st.error("请检查API密钥是否正确，然后重试")
    
    # 如果有初始化错误就停止
    if st.session_state.initialization_error:
        st.error(f"❌ 应用初始化失败，请检查错误: {st.session_state.initialization_error}")
        st.stop()
    
    # 显示文档状态
    if st.session_state.documents_loaded:
        st.success(f"✅ 已加载 {len(st.session_state.documents)} 个文档片段，将使用RAG进行回答")
    else:
        st.warning("⚠️ 未加载任何文档，将直接使用大模型生成回答")
    
    # 辅助函数：将消息列表转换为可读字符串
    def format_chat_history(messages):
        if not messages:
            return ""
        formatted = []
        for msg in messages:
            role = "用户" if msg["role"] == "human" else "助手"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)
    
    # 构建检索函数
    def retrieve_documents(query_input):
        if not query_input.get("chat_history"):
            # 直接使用输入查询
            return st.session_state.search_manager.advanced_search(query_input["input"])
        else:
            # 格式化聊天历史
            chat_history_str = format_chat_history(query_input["chat_history"])
            
            # 使用 LLM 生成的查询
            condensed_query = st.session_state.llm.invoke(
                st.session_state.condense_question_prompt.format(
                    chat_history=chat_history_str,
                    input=query_input["input"]
                )
            ).content
            return st.session_state.search_manager.advanced_search(condensed_query)
    
    # 构建回答生成函数（有文档检索）
    def generate_answer_with_rag(query_and_docs):
        query = query_and_docs["input"]
        docs = query_and_docs["context"]
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # 格式化聊天历史
        chat_history_str = format_chat_history(query_and_docs.get("chat_history", []))
        
        prompt = st.session_state.qa_prompt.format(
            context=context,
            chat_history=chat_history_str,
            input=query
        )
        
        return st.session_state.llm.invoke(prompt).content
    
    # 构建回答生成函数（无文档直接生成）
    def generate_answer_direct(query, chat_history=None):
        # 创建一个简单的提示，直接使用输入，无需检索
        chat_history_str = format_chat_history(chat_history) if chat_history else ""
        
        # 构建系统提示
        system_message = (
            "你是一个基于大语言模型的AI助手。"
            "请根据用户的问题提供有帮助的回答。"
            "如果你不知道答案，请直接说不知道，不要编造信息。"
            f"\n\n聊天历史: {chat_history_str}"
        )
        
        # 直接调用LLM，发送系统消息和用户查询
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=query)
        ]
        
        return st.session_state.llm.invoke(messages).content
    
    # 显示聊天历史
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # 处理用户输入
    if chat_prompt := st.chat_input("请输入您的问题"):
        # 添加用户消息
        st.session_state["chat_messages"].append({"role": "human", "content": chat_prompt})
        with st.chat_message("human"):
            st.write(chat_prompt)
        
        # 生成回答
        with st.chat_message("assistant"):
            try:
                # 检查是否有文档
                if st.session_state.documents_loaded:
                    # 走RAG流程
                    st.info("正在检索相关文档...")
                    # 准备输入
                    query_input = {
                        "input": chat_prompt,
                        "chat_history": st.session_state["chat_messages"][:-1]  # 不包含最新的用户消息
                    }
                    # 检索文档
                    docs = retrieve_documents(query_input)
                    
                    # 显示检索结果
                    with st.expander("查看检索结果", expanded=False):
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**文档 {i}:**")
                            st.markdown(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    
                    # 生成回答
                    st.info("正在生成回答...")
                    query_and_docs = {
                        "input": chat_prompt,
                        "context": docs,
                        "chat_history": st.session_state["chat_messages"][:-1]  # 不包含最新的用户消息
                    }
                    
                    response = generate_answer_with_rag(query_and_docs)
                else:
                    # 直接生成回答
                    st.info("无文档检索，直接生成回答...")
                    response = generate_answer_direct(chat_prompt, st.session_state["chat_messages"][:-1])
                
                # 显示回答
                st.markdown(response)
                
                # 添加助手消息
                st.session_state["chat_messages"].append({"role": "assistant", "content": response})
                
            except Exception as e:
                import traceback
                st.error(f"生成回答时出错: {str(e)}")
                st.code(traceback.format_exc(), language="python")

except Exception as e:
    st.error(f"应用运行时发生错误: {str(e)}")
    import traceback
    st.code(traceback.format_exc(), language="python") 