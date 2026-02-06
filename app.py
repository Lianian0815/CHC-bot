import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import io

# 1. 配置页面
st.set_page_config(page_title="DeepSeek 医药数据助手", layout="wide")
st.title("🤖 DeepSeek 医药销售数据分析专家")

# 2. 配置项
DEEPSEEK_API_KEY = "sk-02f87256f9b74ea78e1bcad39f8541a2"
EXCEL_PATH = "工作簿5.xlsx"

@st.cache_data
def load_all_data():
    try:
        # 读取所有 Sheet
        all_sheets = pd.read_excel(EXCEL_PATH, sheet_name=None)
        return all_sheets
    except Exception as e:
        st.error(f"文件读取失败，请检查路径。错误: {e}")
        return None

all_data_dict = load_all_data()

if all_data_dict:
    # --- 核心改进：为每个 DataFrame 命名，防止 AI 找错表 ---
    # 我们创建一个列表，但会在 Prompt 里明确告诉 AI 索引对应的表名
    df_list = []
    sheet_names = list(all_data_dict.keys())
    for name in sheet_names:
        df_list.append(all_data_dict[name])

    # 3. 初始化 LLM
    llm = ChatOpenAI(
        model='deepseek-chat',
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base='https://api.deepseek.com',
        temperature=0
    )

    # 4. 强化版专属业务 Prompt
    # 告诉 AI：df1 是《四层方案-数据》，df2 是...
    sheet_mapping_info = "\n".join([f"df{i+1} 对应的是工作表：{name}" for i, name in enumerate(sheet_names)])

    custom_prompt = f"""
    你是一个严谨的医药行业分析师。你现在可以访问多个 DataFrame，编号如下：
    {sheet_mapping_info}

    ### 任务步骤：
    Step 1: 识别用户查询对象（通用名、Totalname、药品名等）。
    Step 2: 选择对应 DataFrame：
    - 如果查询 Totalname 销售额 -> 查找包含“四层方案-数据”字样的 DataFrame。
    - 如果查询通用名对应的身体部位 -> 查找包含“四层方案”字样的 DataFrame。
    - 如果查询通用名销售额（未提渠道）-> 查找包含“通用名销售额（医院零售合并）”的 DataFrame。
    - 如果查询通用名且提及其渠道 -> 查找包含“通用名销售额”的 DataFrame。
    - 如果查询具体“药品名称” -> 查找包含“药品名称销售额”的 DataFrame。

    Step 3: 数据检索与防错（非常重要）：
    - **严禁凭记忆编造数字**。必须通过 `python_repl_ast` 运行代码查询。
    - 在执行查询前，请先用 `df.columns` 确认列名。
    - **防止看错行**：请使用 `df[df['列名'].str.contains('关键词', na=False)]` 进行过滤，并列出该行所有字段核对。
    - 如果数据存在多行，必须进行 `sum()` 求和计算。

    ### 约束规则：
    1. 默认返回 2025 年数据。
    2. 回答格式：必须以 'Final Answer:' 开头。
    3. 如果找到数据，请先展示该行的原始数值快照，再给出结论。
    """

    # 创建 Agent
    # 核心改进：增加 handle_parsing_errors 和命名
    agent = create_pandas_dataframe_agent(
        llm,
        df_list,
        verbose=True,
        allow_dangerous_code=True,
        prefix=custom_prompt,
        handle_parsing_errors=True, # 解决分析失败报错
        max_iterations=10           # 给 AI 更多思考轮次
    )

    # 5. 交互界面
    st.success(f"成功加载以下工作表：{', '.join(sheet_names)}")
    query = st.text_input("💬 请输入您的问题：", placeholder="例如：奥司他韦在2025年的总销售额是多少？")

    if query:
        with st.spinner("正在精准检索数据中..."):
            try:
                # 运行 agent
                response = agent.run(query)
                st.markdown("---")
                st.markdown(f"### 💡 AI 专家回答：\n\n{response}")
            except Exception as e:
                # 即使报错，也将错误详情打印出来方便调试
                st.error(f"分析异常，请尝试更换提问方式。详情: {e}")