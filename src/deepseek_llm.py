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
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成回答"""
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
            json=data
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error from DeepSeek API: {response.text}")
        
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
    
    @property
    def _llm_type(self) -> str:
        return "deepseek" 