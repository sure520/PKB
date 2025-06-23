# LangChain 重构规则

## 背景
为了确保使用最新版 LangChain 框架进行项目重构，每次使用智能体模式重构代码时，必须先检查 LangChain 的最新文档，确认相关类或包的存在和用法。

## 规则
1. **文档验证优先**：在重构代码之前，始终使用 `mcp_langgraph-docs-mcp` 工具查看 [LangChain 最新文档](https://python.langchain.com/)，确认所需的类、接口或包是否存在，并了解其最新的使用方法。
2. **遵循最新实践**：根据 LangChain 的官方文档中的[最佳实践指南](https://python.langchain.com/docs/concepts)进行重构，避免依赖过时的实现方式。
3. **工具调用规范**：当涉及到工具（Tools）时，应参考[工具概念页面](https://python.langchain.com/docs/concepts/tools/)和[如何定义自定义工具](https://python.langchain.com/docs/how_to/custom_tools/)，确保工具的定义和使用符合 LangChain 的推荐做法。
4. **异步编程支持**：如果应用场景需要，应充分利用 LangChain 对[异步编程的支持](https://python.langchain.com/docs/concepts/async)，提高应用性能。
5. **回调机制**：利用 LangChain 提供的[回调系统](https://python.langchain.com/docs/concepts/callbacks/)进行日志记录、监控或流式传输事件。
6. **可扩展性设计**：所有组件都应易于扩展，如需自定义功能，应参考 LangChain 的[自定义指南](https://python.langchain.com/docs/how_to/custom_chat_model/)。
7. **版本兼容性**：注意 LangChain 不同版本之间的差异，尤其是从 v0.0 迁移到 LCEL 和 LangGraph 时的[迁移指南](https://python.langchain.com/docs/versions/migrating_chains/)。
8. **测试与调试**：重构过程中，应按照 LangChain 的[测试指南](https://python.langchain.com/docs/concepts/testing/)编写单元测试和集成测试，确保代码质量。
9. **错误处理**：合理处理可能出现的异常情况，参考 LangChain 中关于[错误处理的最佳实践](https://python.langchain.com/docs/how_to/tool_error/)。
10. **文档更新**：在重构过程中，应及时更新相关的开发文档，保持文档与实际代码的一致性。

## 示例
- 如果需要重构聊天模型部分，请首先查阅[聊天模型概念页](https://python.langchain.com/docs/concepts/chat_models/)和[如何创建自定义聊天模型类](https://python.langchain.com/docs/how_to/custom_chat_model/)。
- 如果涉及检索系统，请参考[检索概念页](https://python.langchain.com/docs/concepts/retrieval/)和[如何编写自定义检索器类](https://python.langchain.com/docs/how_to/custom_retriever/)。
- 对于输出解析器，请参照[输出解析器概念页](https://python.langchain.com/docs/concepts/output_parsers/)和[如何编写自定义输出解析器类](https://python.langchain.com/docs/how_to/output_parser_custom/)。

通过遵循上述规则，可以确保项目的重构工作基于最新的 LangChain 文档和技术，从而构建出更加健壮、灵活的应用程序。