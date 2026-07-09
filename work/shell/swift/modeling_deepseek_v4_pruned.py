"""Custom DeepSeek-V4 model that works with DeepseekV4PrunedConfig.

Wraps the stock DeepseekV4ForCausalLM so that transformers'
``AutoModelForCausalLM.from_pretrained()`` can resolve the model class
via the ``auto_map`` entry in config.json, while declaring the correct
``config_class`` to match the custom config.
"""
import sys
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM as _Base
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

# Resolve the DeepseekV4PrunedConfig class that transformers already loaded
# via its dynamic-module system (trust_remote_code).  The module path
# contains a content hash, so we scan sys.modules at import time.
# By the time this modeling file is imported, the config module has
# already been loaded by transformers, so it will be in sys.modules.
_config_cls = DeepseekV4Config  # safe fallback
for _mod_name, _mod in list(sys.modules.items()):
    if _mod is not None and _mod_name.endswith('.configuration_deepseek_v4_pruned'):
        _cls = getattr(_mod, 'DeepseekV4PrunedConfig', None)
        if _cls is not None:
            _config_cls = _cls
            break


class DeepseekV4ForCausalLM(_Base):
    """Thin subclass that accepts DeepseekV4PrunedConfig.

    ``DeepseekV4PrunedConfig`` is a subclass of ``DeepseekV4Config``, so
    all behaviour is identical.  The only purpose of this wrapper is to
    satisfy transformers' ``config_class`` consistency check during
    ``AutoModelForCausalLM.register()``.
    """
    config_class = _config_cls
