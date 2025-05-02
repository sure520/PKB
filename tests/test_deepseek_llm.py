import unittest
from src.deepseek_llm import DeepSeekChat
from langchain_core.messages import BaseMessageChunk

class TestDeepSeekChat(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.llm = DeepSeekChat(
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        )
    
    def test_chat_completion(self):
        """测试聊天完成功能"""
        # 测试简单的对话
        response = self.llm.invoke("你好，请介绍一下你自己。")
        self.assertIsInstance(response, BaseMessageChunk)
        self.assertGreater(len(response.content), 0)
        
    def test_temperature(self):
        """测试温度参数"""
        # 使用不同的温度值测试
        high_temp_llm = DeepSeekChat(temperature=1.0)
        low_temp_llm = DeepSeekChat(temperature=0.1)
        
        # 使用更有创意性的提示词，更容易产生不同的回答
        prompt = "请用一句话描述春天的景色，要有创意和想象力。"
        response1 = high_temp_llm.invoke(prompt)
        response2 = low_temp_llm.invoke(prompt)
        self.assertNotEqual(response1.content, response2.content)
        
    def test_max_tokens(self):
        """测试最大token限制"""
        # 使用较小的max_tokens
        limited_llm = DeepSeekChat(max_tokens=10)
        response = limited_llm.invoke("请写一个长段落。")
        self.assertLessEqual(len(response.content.split()), 10)

if __name__ == "__main__":
    unittest.main() 