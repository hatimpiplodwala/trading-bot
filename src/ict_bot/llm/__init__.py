"""Local LLM layer via Ollama (Qwen3:4b).

v1 = post-trade journal only; the trade-blocking veto is DEFERRED to v2 and is
only justified if fed genuinely new information (news/sentiment). The LLM is
never on the trade path (Gotcha #3).
"""
