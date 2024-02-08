import os
import dotenv
from langchain.agents import AgentType, initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import Tool
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools.python.tool import PythonREPLTool

dotenv.load_dotenv()


def main():
    print("Start...")
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    csv_agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        "episode_info.csv",
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    grand_agent = initialize_agent(
        tools=[
            Tool(
                name="PythonAgent",
                func=python_agent_executor.run,
                description="""useful when you need to transform natural language and write from it python and execute the python code,
                              returning the results of the code execution,
                            DO NOT SEND PYTHON CODE TO THIS TOOL""",
            ),
            Tool(
                name="CSVAgent",
                func=csv_agent.run,
                description="""useful when you need to answer question over episode_info.csv file,
                             takes an input the entire question and returns the answer after running pandas calculations""",
            ),
        ],
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )

    grand_agent.run(
        """Generate 5 QRcode that point to www.google.com and save them in current working directory, \
        you have the qrcode package installed already."""
    )

    grand_agent.run("print seasons ascending order of the number of episodes they have")


if __name__ == "__main__":
    main()
