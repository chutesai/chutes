from typing import List
from chutedk.image.directive.base_image import FROM
from chutedk.image.directive.apt import APT
from chutedk.image.directive.add import ADD
from chutedk.image.directive.generic_run import RUN
from chutedk.image.directive.workdir import WORKDIR
from chutedk.image.directive.env import ENV
from chutedk.image.directive.user import USER
from chutedk.image.directive.maintainer import MAINTAINER
from chutedk.image.directive.entrypoint import ENTRYPOINT


class Image:

    default_base_image = "nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04"

    def __init__(self):
        """
        Semi-useless constructor - don't try to pass args here, use the provided methods.
        """
        self._directives = [
            FROM(self.default_base_image),
        ]

    def __str__(self):
        """String representation."""
        return "\n".join(list(map(str, self._directives)))

    def from_base(self, base_image: str):
        """
        Replace the FROM directive with an updated base image.

        :param base_image: The updated base image to use.
        :type base_image: str

        :return: Updated image.
        :rtype: Image

        """
        self._directives = [FROM(base_image)] + [
            directive
            for directive in self._directives
            if not isinstance(directive, FROM)
        ]
        return self

    def with_maintainer(self, maintainer: str):
        """
        Helper to set the maintainer for an image.
        """
        self._directives += [
            MAINTAINER(maintainer),
        ]
        return self

    def set_user(self, user: str):
        """
        Helper to set the current user within the image.
        """
        self._directives += [
            USER(user),
        ]
        return self

    def set_workdir(self, directory: str):
        """
        Helper to set the current working directory within the image.
        """
        self._directives += [
            WORKDIR(directory),
        ]
        return self

    def with_python(self, version: str = "3.10.15"):
        """
        Helper to install a particular version of python.

        :param version: Python version to install.
        :type version: str

        :return: Updated image.
        :rtype: Image

        """
        bin_suffix = ".".join(version.split(".")[:2])
        current_workdir = (
            [
                directive._args
                for directive in self._directives
                if isinstance(directive, WORKDIR)
            ]
            or ["/root"]
        )[-1]
        self._directives += [
            APT.update(),
            APT.install(
                [
                    "build-essential",
                    "zlib1g-dev",
                    "libncurses5-dev",
                    "libgdbm-dev",
                    "libnss3-dev",
                    "libssl-dev",
                    "libreadline-dev",
                    "libffi-dev",
                    "libsqlite3-dev",
                    "wget",
                    "libbz2-dev",
                    "libexpat1-dev",
                ]
            ),
            WORKDIR("/usr/src"),
            RUN(
                f"wget https://www.python.org/ftp/python/{version}/Python-{version}.tgz"
            ),
            RUN(f"tar -xzf Python-{version}.tgz"),
            WORKDIR(f"/usr/src/Python-{version}"),
            RUN(
                "./configure --enable-optimizations --enable-shared --with-system-expat --with-ensurepip=install --prefix=/opt/python"
            ),
            RUN("make -j"),
            RUN("make altinstall"),
            WORKDIR(current_workdir),
            RUN(f"ln -s /opt/python/bin/pip{bin_suffix} /opt/python/bin/pip"),
            RUN(f"ln -s /opt/python/bin/python{bin_suffix} /opt/python/bin/python"),
            RUN("echo /opt/python/lib >> /etc/ld.so.conf && ldconfig"),
            RUN("rm -rf /usr/src/Python*"),
        ]
        return self

    def apt_install(self, package: str | List[str]):
        """
        Helper to run an apt install.
        """
        self._directives += [
            APT.install(package),
        ]
        return self

    def apt_remove(self, package: str | List[str]):
        """
        Helper to run an apt remove.
        """
        self._directives += [APT.remove(package)]
        return self

    def with_env(self, key: str, value: str):
        """
        Helper to set environment variables.
        """
        self._directives += [
            ENV(key, value),
        ]
        return self

    def run_command(self, command: str):
        """
        Helper to run arbitrary commands.
        """
        self._directives += [
            RUN(command),
        ]
        return self

    def add(self, *args, **kwargs):
        """
        Helper to add files to the image.
        """
        self._directives += [
            ADD(*args, **kwargs),
        ]
        return self

    def with_entrypoint(self, *args):
        """
        Helper to set the image's entrypoint.
        """
        self._directives = [
            directive
            for directive in self._directives
            if not isinstance(directive, ENTRYPOINT)
        ] + [ENTRYPOINT(*args)]
        return self
