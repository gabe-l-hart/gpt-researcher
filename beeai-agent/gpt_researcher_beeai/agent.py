from typing import Any
import logging

from beeai_sdk.providers.agent import Context, Server
from beeai_sdk.schemas.base import Log
from beeai_sdk.schemas.text import TextInput, TextOutput
from gpt_researcher import GPTResearcher
from openinference.instrumentation.openai import OpenAIInstrumentor

from gpt_researcher_beeai.configuration import load_env

OpenAIInstrumentor().instrument()
load_env()  # GPT Researchers uses env variables for configuration

server = Server("researcher-agent")


@server.agent()
async def run_agent(input: TextInput, ctx: Context) -> TextOutput:
    output: TextOutput = TextOutput(text="")

    class CustomLogsHandler:
        async def send_json(self, data: dict[str, Any]) -> None:
            match data.get("type"):
                case "logs":
                    log = Log(
                        message=data.get("output", ""),
                        metadata=data.get("metadata", None),
                    )
                    output.logs.append(log)
                    await ctx.report_agent_run_progress(
                        TextOutput(logs=[None, log], text="")
                    )
                case "report":
                    output.text += data.get("output", "")
                    await ctx.report_agent_run_progress(
                        TextOutput(text=data.get("output", ""))
                    )

    # Input may attach 'documents' as an extra field
    documents = getattr(input, "documents", None)
    researcher = GPTResearcher(
        query=input.text,
        report_type="research_report",
        report_source="hybrid" if documents else "web",
        documents=documents,
        websocket=CustomLogsHandler(),
    )
    # Don't search on disk for documents
    researcher.cfg.doc_path = None
    # Conduct research on the given query
    try:
        await researcher.conduct_research()
        # Write the report
        await researcher.write_report()
        return output
    except Exception as err:
        logging.error("Caught an excption during research!", exc_info=True)
        return ""
