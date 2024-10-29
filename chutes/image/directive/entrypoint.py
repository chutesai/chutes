import json
from typing import List
from chutes.image.directive import BaseDirective, DirectiveType


class ENTRYPOINT(BaseDirective):
    def __init__(self, args: str | List[str]):
        """
        Construct a new entrypoint directive.

        :param args: String or list of strings to build the ENTRYPOINT directive from.
        :type args: str | List[str]

        """
        self._type = DirectiveType.ENTRYPOINT
        command = [args] if isinstance(args, str) else args
        self._args = json.dumps(command)
        self._build_context = []
