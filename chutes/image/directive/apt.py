import re
from typing import List
from chutes.image.directive import BaseDirective, DirectiveType

PACKAGE_RE = re.compile(r"^[a-zA-Z0-9_:\*-]+$")


class APT(BaseDirective):
    def __init__(self):
        """Constructor."""
        self._type = DirectiveType.RUN
        self._build_context = []

    @classmethod
    def update(cls):
        """
        Perform an apt update.

        :return: RUN directive to perform apt updates.
        :rtype: APT

        """
        directive = cls()
        directive._args = "apt-get update"
        return directive

    @classmethod
    def _install_or_remove(cls, package: str | List[str], command: str = "install"):
        """
        Install or remove one or more packages with apt.

        :param package: A single package (str) or list of packages (list of strings) to install/remove.
        :type package: str or List[str]

        :raises AssertionError: Validation assertions.

        :return: RUN directive to perform apt install/remove.
        :rtype: APT

        """
        packages = [package] if isinstance(package, str) else package
        for pkg in packages:
            assert PACKAGE_RE.match(pkg), f"Invalid apt package option: {pkg}"
        directive = cls()
        directive._args = f"apt-get -y {command} " + " ".join(packages)
        return directive

    @classmethod
    def install(cls, package: str | List[str]):
        """
        Install one or more packages with apt -- wrapper around _install_or_remove with command="install"
        """
        return cls._install_or_remove(package, command="install")

    @classmethod
    def remove(cls, package: str | List[str]):
        """
        Remove one or more packages with apt -- wrapper around _install_or_remove with command="remove"
        """
        return cls._install_or_remove(package, command="remove")
