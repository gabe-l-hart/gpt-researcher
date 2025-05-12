import os
from textwrap import dedent
from typing import Any

from acp_sdk import AnyUrl, Link, LinkType, Message, MessagePart, Metadata

from gpt_researcher import GPTResearcher
from gpt_researcher.utils.enum import PromptFamily, ReportSource, ReportType, Tone

from acp_sdk.server import Context, Server

server = Server()


@server.agent(
    metadata=Metadata(
        programming_language="Python",
        links=[
            Link(
                type=LinkType.SOURCE_CODE,
                url=AnyUrl(
                    f"https://github.com/gabe-l-hart/gpt-researcher/blob/{os.getenv('RELEASE_VERSION', 'main')}"
                    "/beeai-agent"
                ),
            )
        ],
        documentation=dedent(
            """\
            TODO---- Rewrite docs for Granite! -------------------------

            The agent is an autonomous system designed to perform detailed research on any specified topic, leveraging both web and local resources. It generates a long, factual report complete with citations, striving to provide unbiased and accurate information. Drawing inspiration from recent advancements in AI-driven research methodologies, the agent addresses common challenges like misinformation and the limits of traditional LLMs, offering robust performance through parallel processing.

            ## How It Works
            The GPT Researcher agent operates by deploying a 'planner' to generate relevant research questions and 'execution' agents to collect information. The system then aggregates these findings into a well-structured report. This approach minimizes biases by cross-referencing multiple sources and focuses on delivering comprehensive insights. It employs a custom infrastructure to ensure rapid and deterministic outcomes, making it suitable for diverse research applications.

            ## Input Parameters
            - **text** (string) – The topic or query for which the research report is to be generated.

            ## Key Features
            - **Comprehensive Research** – Generates detailed reports using information from multiple sources.
            - **Bias Reduction** – Cross-references data from various platforms to minimize misinformation and bias.
            - **High Performance** – Utilizes parallelized processes for efficient and swift report generation.
            - **Customizable** – Offers customization options to tailor research for specific domains or tasks.
            """
        ),
        use_cases=[
            "**Comprehensive Research** – Generates detailed reports using information from multiple sources.",
            "**Bias Reduction** – Cross-references data from various platforms to minimize misinformation and bias.",
            "**High Performance** – Utilizes parallelized processes for efficient and swift report generation.",
            "**Customizable** – Offers customization options to tailor research for specific domains or tasks.",
        ],
        ui={"type": "hands-off", "user_greeting": "What topic do you want to research?"},
        examples={
            "cli": [
                {
                    "command": 'beeai run gpt_researcher "Impact of climate change on global agriculture"',
                    "name": "Conducting Research",
                    "description": "Conducting Research on Climate Change",
                    "processing_steps": [
                        "Initializes task-specific agents to interpret the query",
                        "Generates a series of questions to form an objective opinion on the topic"
                        "Uses a crawler agent to gather and summarize information for each question",
                        "Aggregates and filters these summaries into a final comprehensive report",
                    ],
                },
            ]
        },
        env=[
            {"name": "LLM_MODEL", "description": "Model to use from the specified OpenAI-compatible API."},
            {"name": "LLM_API_BASE", "description": "Base URL for OpenAI-compatible API endpoint"},
            {"name": "LLM_API_KEY", "description": "API key for OpenAI-compatible API endpoint"},
            {"name": "LLM_MODEL_FAST", "description": "Fast model to use from the specified OpenAI-compatible API."},
            {"name": "LLM_MODEL_SMART", "description": "Smart model to use from the specified OpenAI-compatible API."},
            {
                "name": "LLM_MODEL_STRATEGIC",
                "description": "Strategic model to use from the specified OpenAI-compatible API.",
            },
            {"name": "EMBEDDING_MODEL", "description": "Embedding model to use (see GPT Researcher docs for details)"},
            {"name": "REPORT_TYPE", "description": f"Type of report to generate {[v.value for v in ReportType]}"},
            {"name": "TONE", "description": f"Tone of the report {[v.value for v in Tone]}"},
            {"name": "DOC_PATH", "description": "Path to local documents to use as context"},
            {"name": "HYBRID", "description": "If given local documents, also perform web search"},
        ],
    )
)
async def granite_researcher(input: list[Message], context: Context) -> None:
    """
    The agent conducts in-depth local and web research using a language model to generate comprehensive reports with
    citations, aimed at delivering factual, unbiased information.
    """
    os.environ["RETRIEVER"] = "duckduckgo"
    llm_base = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
    os.environ["OPENAI_API_KEY"] = os.getenv("LLM_API_KEY", "dummy")
    model = os.getenv("LLM_MODEL", "granite3.3:8b-beeai")
    os.environ["LLM_MODEL"] = model
    os.environ["FAST_LLM"] = f"openai:{os.getenv('LLM_MODEL_FAST', model)}"
    os.environ["SMART_LLM"] = f"openai:{os.getenv('LLM_MODEL_SMART', model)}"
    os.environ["STRATEGIC_LLM"] = f"openai:{os.getenv('LLM_MODEL_STRATEGIC', model)}"

    # TODO: Figure out how to get the embedding chunking to match Granite's 512
    #   context size limit
    # embedding_model = os.getenv("EMBEDDING_MODEL", "ollama:granite-embedding:278m")
    embedding_model = os.getenv("EMBEDDING_MODEL", "ollama:nomic-embed-text")
    if embedding_model:
        os.environ["EMBEDDING"] = embedding_model

    if llm_base.endswith("/v1"):
        ollama_base = llm_base.rsplit("/", 1)[0]
        openai_base = llm_base
    else:
        ollama_base = llm_base
        openai_base = llm_base.rstrip("/") + "/v1"
    if any(
        model.startswith("ollama:")
        for model in [
            os.environ["LLM_MODEL"],
            os.environ["FAST_LLM"],
            os.environ["SMART_LLM"],
            os.environ["STRATEGIC_LLM"],
            os.getenv("EMBEDDING", ""),
        ]
    ):
        os.environ["OLLAMA_BASE_URL"] = ollama_base
    os.environ["OPENAI_BASE_URL"] = openai_base

    # Customizations for the report type/tone/model
    report_type = os.getenv("REPORT_TYPE", ReportType.ResearchReport.value)
    tone = os.getenv("TONE", Tone.Objective.value)
    prompt_family = PromptFamily.Granite.value if "granite" in model else None

    # Determine the input sources
    hybrid = os.getenv("HYBRID", "").lower() == "true"
    doc_path = os.getenv("DOC_PATH")
    report_source = ReportSource.Web.value
    if doc_path:
        report_source = ReportSource.Local.value if not hybrid else ReportSource.Hybrid.value
    input_documents = getattr(input, "documents", None)

    # Validate configs against enums
    try:
        ReportType(report_type)
    except ValueError as err:
        raise ValueError(f"Error: Invalid REPORT_TYPE {report_type}. Options: {[v.value for v in ReportType]}")
    try:
        Tone(tone)
    except ValueError as err:
        raise ValueError(f"Error: Invalid TONE {tone}. Options: {[v.value for v in Tone]}")

    class CustomLogsHandler:
        async def send_json(self, data: dict[str, Any]) -> None:
            if "output" not in data:
                return
            match data.get("type"):
                case "logs":
                    await context.yield_async({"message": f"{data['output']}\n"})
                case "report":
                    await context.yield_async(MessagePart(content=data["output"]))

    researcher = GPTResearcher(
        query=str(input[-1]),
        report_type=report_type,
        tone=tone,
        prompt_family=prompt_family,
        report_source=report_source,
        documents=input_documents,
        websocket=CustomLogsHandler(),
    )
    await researcher.conduct_research()
    await researcher.write_report()


def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()
