# gpt-researcher-beeai
beeai agent wrapper for gpt-researcher



## Environment Variables

 - `PLATFORM_URL` - The url for the acp endpoint on the beeai server e.g. `http://127.0.0.1:8333`

```yaml
  - name: LLM_MODEL
    required: false
    description: "Model to use from the specified OpenAI-compatible API."
  - name: LLM_API_BASE
    required: false
    description: "Base URL for OpenAI-compatible API endpoint"
  - name: LLM_API_KEY
    required: false
    description: "API key for OpenAI-compatible API endpoint"
  - name: LLM_MODEL_FAST
    required: false
    description: "Fast model to use from the specified OpenAI-compatible API."
  - name: LLM_MODEL_SMART
    required: false
    description: "Smart model to use from the specified OpenAI-compatible API."
  - name: LLM_MODEL_STRATEGIC
    required: false
    description: "Strategic model to use from the specified OpenAI-compatible API."
  - name: EMBEDDING_MODEL
    required: false
    description: "Embedding model to use (see GPT Researcher docs for details)"
```
