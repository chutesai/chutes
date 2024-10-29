from chutes.image.directive import BaseDirective, DirectiveType


class MAINTAINER(BaseDirective):
    def __init__(self, maintainer: str):
        """
        Construct a MAINTAINER directive.

        :param maintainer: The purported maintainer of the image.
        :type maintainer: str

        """
        self._type = DirectiveType.MAINTAINER
        self._args = maintainer
        self._build_context = []
