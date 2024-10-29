import re
from chutes.image.directive import BaseDirective, DirectiveType

ENV_KEY_RE = re.compile(r"^([a-z_][a-z0-9_]*)$", re.I)


class ENV(BaseDirective):
    def __init__(self, key: str, value: str):
        """
        Construct a new ENV directive.

        :param key: The environment variable name.
        :type key: str

        :param value: The value to set the variable to.
        :type value: str

        :raises AssertionError: Validation assertions.

        """
        assert ENV_KEY_RE.match(key), f"Invalid variable name: {key}"
        self._type = DirectiveType.ENV
        self._args = f"{key}={value}"
        self._build_context = []
