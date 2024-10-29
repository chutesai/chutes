from chutes.image.directive import BaseDirective, DirectiveType


class RUN(BaseDirective):
    def __init__(self, command: str):
        """
        Construct a new, arbitrary run command - danger zone!

        :param command: The command to run.
        :type command: str

        """
        self._type = DirectiveType.RUN
        self._args = command
        self._build_context = []
