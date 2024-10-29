import re
from chutes.image.directive import BaseDirective, DirectiveType

IMAGE_RE = re.compile(
    r"^((([a-z0-9.-]+)(:[0-9]+)?/)?[a-z0-9._-]+(/[a-z0-9._-]+)*)(:[\w.-]+)?(@sha256:[a-f0-9]{64})?$",
    re.I,
)


class FROM(BaseDirective):
    def __init__(self, image):
        """
        Construct a new FROM directive.

        :param image: The image to build on top of.
        :type image: str

        :raises AssertionError: Validation assertions.

        """
        assert IMAGE_RE.match(image), f"Invalid image: {image}"
        self._type = DirectiveType.FROM
        self._args = image
        self._build_context = []
