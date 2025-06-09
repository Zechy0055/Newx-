"""Handles interactions with Qwen series of models by Alibaba Cloud.

This module will contain the necessary logic to make API calls to Qwen,
handle authentication, and process responses.
"""

import os
import json
import requests # Placeholder for actual API call library if different

from app.model.common import LiteLLMGeneric # Import the base class

# Environment variable for TogetherAI API Key, which LiteLLM will use for TogetherAI models
TOGETHERAI_API_KEY_ENV_VAR = "TOGETHERAI_API_KEY"

# Qwen models are often accessed via TogetherAI or other LiteLLM providers.
# The model name will be like "together_ai/qwen/qwen-turbo"

class QwenModel(LiteLLMGeneric):
    """
    Wrapper class for interacting with Qwen language models via LiteLLM,
    typically using TogetherAI as the provider.

    This class inherits from `LiteLLMGeneric` and handles Qwen-specific
    configurations if any, primarily API key checking for TogetherAI.
    The actual API calls are handled by the `LiteLLMGeneric.call()` method.
    """
    def __init__(self, model_name: str, cost_per_input: float, cost_per_output: float, parallel_tool_call: bool = False):
        """
        Initializes the QwenModel.

        Args:
            model_name (str): The specific Qwen model name as recognized by LiteLLM via TogetherAI
                              (e.g., "together_ai/qwen/qwen-1.8b-chat").
            cost_per_input (float): Cost per input token for this model (placeholder).
            cost_per_output (float): Cost per output token for this model (placeholder).
            parallel_tool_call (bool): Whether the model supports parallel tool calls. Defaults to False.
        """
        super().__init__(model_name, cost_per_input, cost_per_output, parallel_tool_call)
        # self.name is set by super().__init__ to model_name
        self.api_key = self.check_api_key() # Check for key on init

        if not self.api_key:
            # LiteLLMGeneric.call will ultimately fail if the key is truly needed and not set
            # in the environment where LiteLLM runs. This is more of an early warning.
            print(f"Warning: TogetherAI API key ({TOGETHERAI_API_KEY_ENV_VAR}) not found for Qwen model {self.name}. API calls may fail.")

    def check_api_key(self) -> str:
        """
        Checks for the TogetherAI API key in environment variables.

        Returns:
            str: The API key if found, otherwise an empty string.
        """
        api_key = os.getenv(TOGETHERAI_API_KEY_ENV_VAR)
        if not api_key:
            # print(f"Warning: Environment variable {TOGETHERAI_API_KEY_ENV_VAR} not set.") # Optional warning
            return ""
        return api_key

    def setup(self) -> None:
        """
        Performs any setup required for the Qwen model via LiteLLM.
        Currently, this relies on the base class setup and API key check.
        """
        super().setup() # Calls LiteLLMGeneric's setup (which is a pass for now)
        if not self.api_key:
             # This warning is more for the user running the app, LiteLLM will handle actual error if key missing
            print(f"QwenModel ({self.name}): API key from {TOGETHERAI_API_KEY_ENV_VAR} is missing. Ensure it's set for LiteLLM/TogetherAI calls.")

    # No need to implement `call` or `get_overall_exec_stats` as they are inherited from LiteLLMGeneric.
    # `generate_raw_response` is removed as `call` is the standard interface from Model.

# The standalone call_qwen function might be deprecated or significantly changed if all interactions
# are to go through the LiteLLMGeneric interface via the QwenModel class.
# If a direct, non-LiteLLM call to Qwen (e.g., via DashScope) is still needed, this function would
# need its own implementation. For now, we'll remove it to align with LiteLLM usage.

# def call_qwen(prompt: str, model_name: str = "qwen-turbo", api_key: str | None = None, **kwargs) -> str:
#     """
#     Makes a direct call to a Qwen model API (e.g., DashScope).
#     This function is DEPRECATED if using QwenModel with LiteLLM/TogetherAI.
#     """
#     raise NotImplementedError("Direct call_qwen function is deprecated or needs specific implementation for non-LiteLLM Qwen access.")


if __name__ == "__main__":
    # This example assumes QwenModel will be used via LiteLLM and TogetherAI
    print(f"To use Qwen models (e.g., 'together_ai/qwen/qwen-1.8b-chat') via this wrapper:")
    print(f"1. Ensure LiteLLM is configured (it's part of the project).")
    print(f"2. Set the {TOGETHERAI_API_KEY_ENV_VAR} environment variable with your TogetherAI API key.")
    print(f"3. Register instances of QwenModel in app/model/register.py with appropriate model names.")

    # Example of how it might be instantiated (costs are placeholders)
    # test_qwen_model = QwenModel(
    #     model_name="together_ai/qwen/qwen-1.8b-chat", # Example LiteLLM/TogetherAI name
    #     cost_per_input=0.0,
    #     cost_per_output=0.0
    # )
    # print(f"QwenModel instance for {test_qwen_model.name} created.")
    # if test_qwen_model.api_key:
    #     print(f"Found TogetherAI API key.")
    # else:
    #     print(f"Warning: {TOGETHERAI_API_KEY_ENV_VAR} not found.")

    # To test the call (would require LiteLLM setup and valid key):
    # try:
    #     response_content, _, _, _ = test_qwen_model.call(messages=[{"role": "user", "content": "Hello!"}])
    #     print(f"Response from {test_qwen_model.name}: {response_content}")
    # except Exception as e:
    #     print(f"Error calling Qwen model via LiteLLM: {e}")
    pass
