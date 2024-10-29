from enum import Enum


class DirectiveType(Enum):
    """Directives within Dockerfile that we currently support."""

    FROM = "FROM"
    ADD = "ADD"
    ENTRYPOINT = "ENTRYPOINT"
    ENV = "ENV"
    MAINTAINER = "MAINTAINER"
    RUN = "RUN"
    USER = "USER"
    WORKDIR = "WORKDIR"


class BaseDirective:
    def __init__(self, _type: DirectiveType, _args: str):
        """
        Constructor.  Nothing fancy here, use the derived classes instead.
        """
        self._type = _type
        self._args = _args
        self._build_context = []

    def __str__(self):
        """
        String representation.
        """
        return f"{self._type.value} {self._args}"
