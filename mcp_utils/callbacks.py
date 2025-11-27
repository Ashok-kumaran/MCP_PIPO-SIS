# mcp_utils/callbacks.py
import logging
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("pipo.callbacks")

class LogEverythingCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        logger.info("[LLM_START] prompts=%s", prompts)

    def on_llm_end(self, response, **kwargs):
        logger.info("[LLM_END] response=%s", response)

    def on_llm_error(self, error, **kwargs):
        logger.exception("[LLM_ERROR] %s", error)

    def on_chain_start(self, serialized, inputs, **kwargs):
        logger.info("[CHAIN_START] inputs=%s", inputs)

    def on_chain_end(self, outputs, **kwargs):
        logger.info("[CHAIN_END] outputs=%s", outputs)

    def on_chain_error(self, error, **kwargs):
        logger.exception("[CHAIN_ERROR] %s", error)

    def on_tool_start(self, tool, input, **kwargs):
        logger.info("[TOOL_START] %s %s", tool, input)

    def on_tool_end(self, output, **kwargs):
        logger.info("[TOOL_END] output=%s", output)

    def on_tool_error(self, error, **kwargs):
        logger.exception("[TOOL_ERROR] %s", error)

    def on_agent_action(self, action, **kwargs):
        logger.info("[AGENT_ACTION] %s", action)

    def on_agent_finish(self, finish, **kwargs):
        logger.info("[AGENT_FINISH] %s", finish)
