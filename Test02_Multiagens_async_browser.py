'''
多智能体协作实现浏览器自动化\
! playwright install 会下载并安装 Playwright 支持的浏览器内核
'''
from ast import main
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv 
load_dotenv(override=True) 
DeepSeek_API_KEY = os.getenv("DeepSeek_API_KEY")
# print(DeepSeek_API_KEY)

# 初始化 PlayWright 浏览器
sync_browser = create_sync_playwright_browser()
# print(sync_browser)
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
# print(toolkit)
tools = toolkit.get_tools()
# print(tools)

# 通过langchain hub 拉取提示词模板
# 从官方仓库拉一份已经调好的 OpenAI-tools 专用提示词”，省去手写 prompt 的麻烦，后续直接喂给 create_openai_tools_agent(..., prompt=prompt) 即可。
# 系统人设（“你是一个有帮助的助手”）
# 必须的两个占位符 {input}、{agent_scratchpad}
# 可选的 {chat_history} 占位符
# 因此拿来即可用，无需自己再拼提示
prompt = hub.pull("hwchase17/openai-tools-agent")

# 初始化模型
model = init_chat_model("deepseek-chat", model_provider="deepseek")

# 通过langchain 创建 OpenAI 工具代理
agent = create_openai_tools_agent(model, tools, prompt)

# 通过AgentExecutor 执行代理
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == '__main__':
    # 定义任务
    command = {
      "input": "访问豆瓣电影排行榜:https://movie.douban.com/chart, 提取前5部电影的名称、评分、概况、图片链接",
    }
    # 执行任务
    response = agent_executor.invoke(command)
    print(response)

