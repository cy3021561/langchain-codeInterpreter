import os
import dotenv
from langchain.agents import AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
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
    python_agent_executor.run(
        """Generate 5 QRcode that point to www.yahoo.com and save them in current working directory, \
        you have the qrcode package installed already."""
    )


if __name__ == "__main__":
    main()
