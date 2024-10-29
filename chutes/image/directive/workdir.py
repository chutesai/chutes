from chutes.image.directive import BaseDirective, DirectiveType


class WORKDIR(BaseDirective):
    def __init__(self, workdir: str):
        """
        Construct a WORKDIR directive.

        :param workdir: Working directory to use.
        :type workdir: str

        """
        self._type = DirectiveType.WORKDIR
        self._args = workdir
        self._build_context = []
