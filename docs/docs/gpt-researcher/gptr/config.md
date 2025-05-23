# Configuration

The config.py enables you to customize GPT Researcher to your specific needs and preferences.

Thanks to our amazing community and contributions, GPT Researcher supports multiple LLMs and Retrievers.
In addition, GPT Researcher can be tailored to various report formats (such as APA), word count, research iterations depth, etc.

GPT Researcher defaults to our recommended suite of integrations: [OpenAI](https://platform.openai.com/docs/overview) for LLM calls and [Tavily API](https://app.tavily.com) for retrieving real-time web information.

As seen below, OpenAI still stands as the superior LLM. We assume it will stay this way for some time, and that prices will only continue to decrease, while performance and speed increase over time.

<div style={{ marginBottom: '10px' }}>
<img align="center" height="350" src="/img/leaderboard.png" />
</div>

The default config.py file can be found in `/gpt_researcher/config/`. It supports various options for customizing GPT Researcher to your needs.
You can also include your own external JSON file `config.json` by adding the path in the `config_file` param. **Please follow the config.py file for additional future support**.

Below is a list of current supported options:

- **`RETRIEVER`**: Web search engine used for retrieving sources. Defaults to `tavily`. Options: `duckduckgo`, `bing`, `google`, `searchapi`, `serper`, `searx`. [Check here](https://github.com/assafelovic/gpt-researcher/tree/master/gpt_researcher/retrievers) for supported retrievers
- **`EMBEDDING`**: Embedding model. Defaults to `openai:text-embedding-3-small`. Options: `ollama`, `huggingface`, `azure_openai`, `custom`.
- **`FAST_LLM`**: Model name for fast LLM operations such summaries. Defaults to `openai:gpt-4o-mini`.
- **`SMART_LLM`**: Model name for smart operations like generating research reports and reasoning. Defaults to `openai:gpt-4o`.
- **`STRATEGIC_LLM`**: Model name for strategic operations like generating research plans and strategies. Defaults to `openai:o1-preview`.
- **`LANGUAGE`**: Language to be used for the final research report. Defaults to `english`.
- **`CURATE_SOURCES`**: Whether to curate sources for research. This step adds an LLM run which may increase costs and total run time but improves quality of source selection. Defaults to `True`.
- **`FAST_TOKEN_LIMIT`**: Maximum token limit for fast LLM responses. Defaults to `2000`.
- **`SMART_TOKEN_LIMIT`**: Maximum token limit for smart LLM responses. Defaults to `4000`.
- **`STRATEGIC_TOKEN_LIMIT`**: Maximum token limit for strategic LLM responses. Defaults to `4000`.
- **`BROWSE_CHUNK_MAX_LENGTH`**: Maximum length of text chunks to browse in web sources. Defaults to `8192`.
- **`SUMMARY_TOKEN_LIMIT`**: Maximum token limit for generating summaries. Defaults to `700`.
- **`TEMPERATURE`**: Sampling temperature for LLM responses, typically between 0 and 1. A higher value results in more randomness and creativity, while a lower value results in more focused and deterministic responses. Defaults to `0.55`.
- **`TOTAL_WORDS`**: Total word count limit for document generation or processing tasks. Defaults to `800`.
- **`REPORT_FORMAT`**: Preferred format for report generation. Defaults to `APA`. Consider formats like `MLA`, `CMS`, `Harvard style`, `IEEE`, etc.
- **`MAX_ITERATIONS`**: Maximum number of iterations for processes like query expansion or search refinement. Defaults to `3`.
- **`AGENT_ROLE`**: Role of the agent. This might be used to customize the behavior of the agent based on its assigned roles. No default value.
- **`MAX_SUBTOPICS`**: Maximum number of subtopics to generate or consider. Defaults to `3`.
- **`SCRAPER`**: Web scraper to use for gathering information. Defaults to `bs` (BeautifulSoup). You can also use [newspaper](https://github.com/codelucas/newspaper).
- **`MAX_SCRAPER_WORKERS`**: Maximum number of concurrent scraper workers per research. Defaults to `15`.
- **`DOC_PATH`**: Path to read and research local documents. Defaults to an empty string indicating no path specified.
- **`PROMPT_FAMILY`**: The family of prompts and prompt formatting to use. Defaults to prompting optimized for GPT models. See the full list of options in [enum.py](https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/utils/enum.py#L56).
- **`LLM_KWARGS`**: Json formatted dict of additional keyword args to be passed to the LLM provider class when instantiating it. This is primarily useful for clients like Ollama that allow for additional keyword arguments such as `num_ctx` that influence the inference calls.
- **`EMBEDDING_KWARGS`**: Json formatted dict of additional keyword args to be passed to the embedding provider class when instantiating it.
- **`USER_AGENT`**: Custom User-Agent string for web crawling and web requests.
- **`MEMORY_BACKEND`**: Backend used for memory operations, such as local storage of temporary data. Defaults to `local`.

To change the default configurations, you can simply add env variables to your `.env` file as named above or export manually in your local project directory.

For example, to manually change the search engine and report format:
```bash
export RETRIEVER=bing
export REPORT_FORMAT=IEEE
```
Please note that you might need to export additional env vars and obtain API keys for other supported search retrievers and LLM providers. Please follow your console logs for further assistance.
To learn more about additional LLM support you can check out the docs [here](/docs/gpt-researcher/llms/llms).

You can also include your own external JSON file `config.json` by adding the path in the `config_file` param.

