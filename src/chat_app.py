import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from vector_store import VectorStore
from zhipuai_embedding import ZhipuAIEmbeddings
from deepseek_llm import DeepSeekChat
from search_manager import SearchManager
from datetime import datetime, timedelta

# 加载环境变量
load_dotenv()

def get_search_manager():
    """获取搜索管理器"""
    embedding = ZhipuAIEmbeddings()
    vector_store = VectorStore(
        embedding=embedding,
        persist_directory="../vector_db"
    )
    vector_store.load_existing()
    return SearchManager(vector_store)

def get_qa_history_chain(search_manager):
    """构建问答链"""
    llm = DeepSeekChat(
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=2000
    )
    
    # 问题总结模板
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    # 构建检索分支
    retrieve_docs = RunnableBranch(
        (
            lambda x: not x.get("chat_history"), 
            lambda x: search_manager.advanced_search(x["input"])
        ),
        (
            lambda x: True,
            RunnablePassthrough.assign(
            condensed_query=condense_question_prompt 
            | llm 
            | StrOutputParser()
        ) | (lambda x: search_manager.advanced_search(x["condensed_query"]))
        )
    )

    # 回答生成模板
    system_prompt = (
        "你是一个基于知识库的问答助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    # 构建问答链
    qa_chain = (
        RunnablePassthrough().assign(context=lambda x: "\n\n".join(doc.page_content for doc in x["context"]))
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return RunnablePassthrough().assign(
        context=retrieve_docs,
    ).assign(answer=qa_chain)

def gen_response(chain, input, chat_history):
    """生成回答"""
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    full_response = ""
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]
            full_response += res["answer"]  # 拼接完整回答
    return full_response  # 返回完整回答

def main():
    """主函数"""
    st.set_page_config(
        page_title="个人知识库助手",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 个人知识库助手")
    st.markdown("""
    这是一个基于大语言模型的个人知识库助手，可以回答您关于知识库内容的问题。
    请在下方的输入框中输入您的问题。
    """)

    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "search_manager" not in st.session_state:
        st.session_state.search_manager = get_search_manager()
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain(st.session_state.search_manager)

    # 创建侧边栏
    with st.sidebar:
        st.header("搜索设置")
        
        # 相似度阈值设置
        score_threshold = st.slider(
            "相似度阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        # 日期范围过滤
        st.subheader("日期范围")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=30)
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now()
            )
            
        # 搜索历史
        st.subheader("搜索历史")
        history = st.session_state.search_manager.get_search_history()
        for item in history:
            with st.expander(f"{item['query']} ({item['timestamp'].strftime('%Y-%m-%d %H:%M')})"):
                st.write(f"结果数量: {item['result_count']}")
                if item['filters']:
                    st.write("过滤条件:", item['filters'])
                    
        if st.button("清空搜索历史"):
            st.session_state.search_manager.clear_search_history()
            st.success("搜索历史已清空！")  # 添加提示信息

    # 将相似度阈值和日期范围传递给搜索管理器
    filters = {
        "score_threshold": score_threshold,
        "date_range": (start_date, end_date)
    }
    st.session_state.search_manager.set_filters(filters)  # 假设 SearchManager 支持设置过滤条件

    # 创建主界面
    messages = st.container(height=550)
    
    # 显示历史消息
    for message in st.session_state.messages:
        with messages.chat_message(message[0]):
            st.write(message[1])

    # 处理用户输入
    if prompt := st.chat_input("请输入您的问题"):
        # 添加用户消息
        st.session_state.messages.append(("human", prompt))
        with messages.chat_message("human"):
            st.write(prompt)

        # 获取相似的历史查询
        similar_queries = st.session_state.search_manager.get_similar_queries(prompt)
        if similar_queries:
            with st.expander("相似的历史查询"):
                for query in similar_queries:
                    st.write(f"- {query}")

        # 生成回答
        with messages.chat_message("assistant"):
            answer_generator = gen_response(
                chain=st.session_state.qa_history_chain,
                input=prompt,
                chat_history=st.session_state.messages
            )
            output = "".join(st.write_stream(answer_generator))  # 拼接完整回答
        
        # 保存助手回答
        st.session_state.messages.append(("assistant", output))

if __name__ == "__main__":
    main()
