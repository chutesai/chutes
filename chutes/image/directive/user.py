from chutes.image.directive import BaseDirective, DirectiveType


class USER(BaseDirective):
    def __init__(self, user: str):
        """
        Construct a USER directive.

        :param user: The user to set docker context to.
        :type user: str

        """
        self._type = DirectiveType.USER
        self._args = user
        self._build_context = []
