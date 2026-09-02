# GenAI/RAG Extension

This extension shows how generative AI could support users of rainfall and climate-scenario products.

The current implementation is intentionally local and lightweight. It does not call an external LLM. Instead, it demonstrates the retrieval part of a RAG workflow by searching project documents and product summaries, then returning cited evidence snippets.

## Why This Belongs In The Project

Climate-scenario products need more than model output. Users also need help interpreting indicators, uncertainty ranges, and limitations. A RAG assistant can support that by grounding answers in approved documentation and generated product summaries.

## Included Components

- `src/meteo_ml/rag.py`: local TF-IDF retrieval and extractive answer formatting
- `scripts/query_scenario_assistant.py`: command-line question answering demo
- `scripts/plot_rag_workflow.py`: RAG workflow diagram
- `figures/rag_scenario_assistant_workflow.png`: user-facing workflow figure

## Example

```bash
python3 scripts/query_scenario_assistant.py "How should users interpret heavy-rain scenario indicators?"
```

The answer includes retrieved evidence and source filenames. A production version could add embeddings, an LLM response layer, evaluation, and access control.
