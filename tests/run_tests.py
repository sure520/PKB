import unittest
import os
import sys
from dotenv import load_dotenv

def run_tests():
    """运行所有测试"""
    # 加载环境变量
    load_dotenv()
    
    # 获取测试目录
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 添加项目根目录到 Python 路径
    project_root = os.path.dirname(test_dir)
    sys.path.insert(0, project_root)
    
    # 发现并运行所有测试
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回测试结果
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 