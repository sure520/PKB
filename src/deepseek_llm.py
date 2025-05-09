import os
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field
import requests

class DeepSeekChat(BaseChatModel):
    """DeepSeek聊天模型封装"""
    
    api_key: str = Field(default="")
    model: str = Field(default="deepseek-chat")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)
    api_base: str = Field(default="https://api.deepseek.com")
    demo_mode: bool = Field(default=False)  # 演示模式
    
    def __init__(self, **kwargs):
        # 处理demo_mode参数
        demo_mode = kwargs.pop("demo_mode", False)
        super().__init__(**kwargs)
        
        # 先尝试从参数中获取 API 密钥
        self.api_key = kwargs.get("api_key", "")
        
        # 如果参数中没有提供，从环境变量获取
        if not self.api_key:
            self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
            # 如果 os.getenv 没找到，尝试直接从 .env 文件读取
            if not self.api_key:
                try:
                    env_path = os.path.join(os.getcwd(), '.env')
                    if os.path.exists(env_path):
                        with open(env_path, 'r') as f:
                            for line in f:
                                if line.startswith('DEEPSEEK_API_KEY='):
                                    self.api_key = line.split('=', 1)[1].strip()
                                    if self.api_key.startswith('"') and self.api_key.endswith('"'):
                                        self.api_key = self.api_key[1:-1]
                                    break
                except Exception as e:
                    print(f"读取 .env 文件出错: {e}")
        
        # 检测是否为测试密钥
        self.demo_mode = demo_mode or not self.api_key or self.api_key.startswith("test_key_")
        
        # 输出调试信息
        print(f"DeepSeekChat 初始化: API密钥长度 = {len(self.api_key) if self.api_key else 0}, 演示模式 = {self.demo_mode}")
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成回答"""
        # 演示模式下返回模拟响应
        if self.demo_mode:
            return self._generate_demo_response(messages)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 转换消息格式
            formatted_messages = []
            for message in messages:
                if message.type == "human":
                    formatted_messages.append({"role": "user", "content": message.content})
                elif message.type == "system":
                    formatted_messages.append({"role": "system", "content": message.content})
                elif message.type == "ai":
                    formatted_messages.append({"role": "assistant", "content": message.content})
            
            data = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            if stop:
                data["stop"] = stop
            
            response = requests.post(
                f"{self.api_base}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30  # 添加超时设置
            )
            
            if response.status_code != 200:
                error_msg = f"DeepSeek API 错误: 状态码 {response.status_code}, 响应: {response.text}"
                print(error_msg)
                return self._generate_error_response(error_msg, messages)
            
            result = response.json()
            message = result["choices"][0]["message"]
            
            return ChatResult(
                generations=[
                    ChatGenerationChunk(
                        message=AIMessage(content=message["content"]),
                        text=message["content"]
                    )
                ]
            )
        except Exception as e:
            error_msg = f"API调用出错: {str(e)}"
            print(error_msg)
            return self._generate_error_response(error_msg, messages)
    
    def _generate_demo_response(self, messages: List[BaseMessage]) -> ChatResult:
        """在演示模式下生成模拟回复"""
        # 获取最后一条用户消息
        last_message = messages[-1].content if messages and messages[-1].type == "human" else "你好"
        
        try:
            # 检测是否包含上下文信息，如果有则生成更有意义的回复
            system_message = next((msg for msg in messages if msg.type == "system"), None)
            if system_message and "上下文" in system_message.content:
                # 从系统消息中提取上下文
                context_start = system_message.content.find("上下文:") + 4
                context_end = system_message.content.find("\n\n", context_start) if "\n\n" in system_message.content[context_start:] else len(system_message.content)
                context = system_message.content[context_start:context_end].strip()
                
                # 根据上下文生成响应
                if "对不起，我无法找到与您问题相关的信息" in context or "无法找到" in context:
                    response_text = f"对不起，我在知识库中找不到关于'{last_message}'的相关信息。请尝试询问其他问题，或使用更具体的描述。"
                else:
                    # 生成一个基于上下文的简单回复
                    response_text = f"根据我的知识库信息，关于'{last_message}'：\n\n{context}\n\n请注意，这是演示模式的回复，如需更准确的回答，请配置DeepSeek API密钥。"
            else:
                # 生成简单的模拟回复
                response_text = f"这是演示模式的回复。您问了：'{last_message}'。由于没有配置DeepSeek API密钥，我无法提供实际的AI回复。请添加DEEPSEEK_API_KEY以启用完整功能。"
        except Exception as e:
            # 出错时提供一个回退响应
            response_text = f"这是一个演示模式的回复。您的问题是：'{last_message}'。我们遇到了一些技术问题：{str(e)}。请配置API密钥以获得更好的体验。"
        
        return ChatResult(
            generations=[
                ChatGenerationChunk(
                    message=AIMessage(content=response_text),
                    text=response_text
                )
            ]
        )
    
    def _generate_error_response(self, error_msg: str, messages: List[BaseMessage]) -> ChatResult:
        """生成错误响应"""
        last_message = messages[-1].content if messages and messages[-1].type == "human" else "你好"
        response_text = f"调用DeepSeek API时发生错误: {error_msg}\n\n您的问题是: '{last_message}'。请检查API密钥和网络连接。"
        
        return ChatResult(
            generations=[
                ChatGenerationChunk(
                    message=AIMessage(content=response_text),
                    text=response_text
                )
            ]
        )
    
    @property
    def _llm_type(self) -> str:
        return "deepseek" 