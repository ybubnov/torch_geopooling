import os
import shutil
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path
from typing import List

from conan.api.conan_api import ConanAPI
from conan.api.model import Remote
from conan.api.output import ConanOutput, cli_out_write
from conan.cli.printers import print_profiles
from conan.cli.printers.graph import print_graph_basic, print_graph_packages
from conans.client.graph.graph import CONTEXT_HOST, RECIPE_CONSUMER, Node
from conans.client.graph.graph_builder import DepsGraphBuilder
from conans.client.graph.profile_node_definer import consumer_definer
from conans.model.conan_file import ConanFile
from conans.model.requires import Requirements
from conans.util.files import save
from dotenv.main import DotEnv

if sys.version_info >= (3, 11):
    import tomllib
else:
    import toml as tomllib


class BuildExtBackend:
    def __init__(
        self,
        pyproject_path: Path = Path("pyproject.toml"),
        profile_name: str = "default",
    ) -> None:
        self.profile_name = profile_name
        self.conan_api = ConanAPI()

        self.initialize_configuration(pyproject_path.read_text())
        self.initialize_profiles()
        self.initialize_conanfile()

        self.output_folder = tempfile.mkdtemp()

    @property
    def source_folder(self) -> str:
        return str(Path.cwd())

    @property
    def remotes(self) -> List[Remote]:
        remotes = []

        for r in self.configuration.get("remotes", []):
            remote = Remote(
                r["name"],
                r["url"],
                r["verify_ssl"],
                r.get("disabled", False),
                r.get("allowed_packages"),
                r.get("remote_type"),
            )
            remotes.append(remote)
        return remotes

    def initialize_configuration(self, pyproject_text: str) -> None:
        pyproject = tomllib.loads(pyproject_text)
        self.configuration = pyproject.get("tool", {}).get("conan")

    def initialize_conanfile(self) -> None:
        conanfile = ConanFile("torch_geopooling")

        conanfile.requires = Requirements(
            declared=self.configuration.get("requires"),
            declared_build_tool=self.configuration.get("tool_requires"),
        )
        conanfile.generators = self.configuration.get("generators", [])
        consumer_definer(conanfile, self.profile_host, self.profile_build)

        self.conanfile = conanfile

    def detect_profile(self) -> None:
        profile_pathname = self.conan_api.profiles.get_path(
            self.profile_name, os.getcwd(), exists=False
        )
        detected_profile = self.conan_api.profiles.detect()

        ConanOutput().success("\nDetected profile:")
        cli_out_write(detected_profile.dumps())

        contents = detected_profile.dumps()
        save(profile_pathname, contents)

    def initialize_profiles(self) -> None:
        build_profiles = [self.conan_api.profiles.get_default_build()]
        host_profiles = [self.conan_api.profiles.get_default_host()]

        global_conf = self.conan_api.config.global_conf
        global_conf.validate()  # TODO: Remove this from here

        cache_settings = self.conan_api.config.settings_yml
        profile_plugin = self.conan_api.profiles._load_profile_plugin()
        cwd = str(Path.cwd())

        settings_build = None
        options_build = None
        conf_build = None

        profile_build = self.conan_api.profiles._get_profile(
            build_profiles,
            settings_build,
            options_build,
            conf_build,
            cwd,
            cache_settings,
            profile_plugin,
            global_conf,
        )

        settings_host = None
        options_host = None
        conf_host = None

        profile_host = self.conan_api.profiles._get_profile(
            host_profiles,
            settings_host,
            options_host,
            conf_host,
            cwd,
            cache_settings,
            profile_plugin,
            global_conf,
        )

        self.profile_host = profile_host
        self.profile_build = profile_build

    def make_dependency_graph(self) -> DepsGraphBuilder:
        root_node = Node(None, self.conanfile, context=CONTEXT_HOST, recipe=RECIPE_CONSUMER)

        return self.conan_api.graph.load_graph(
            root_node,
            profile_host=self.profile_host,
            profile_build=self.profile_build,
            remotes=self.remotes,
        )

    def install(self) -> None:
        print_profiles(self.profile_host, self.profile_build)

        graph = self.make_dependency_graph()
        print_graph_basic(graph)

        graph.report_graph_error()
        self.conan_api.graph.analyze_binaries(graph, remotes=self.remotes, build_mode=["missing"])

        print_graph_packages(graph)
        self.conan_api.install.install_binaries(deps_graph=graph, remotes=self.remotes)

        ConanOutput().title("Finalizing install (deploy, generators)")
        self.conan_api.install.install_consumer(
            graph, source_folder=self.source_folder, output_folder=self.output_folder
        )

        ConanOutput().success("Install finished successfully")

    def source(self) -> None:
        process = subprocess.run(
            [f"source {self.output_folder}/conanbuild.sh && env"],
            shell=True,
            check=True,
            env=os.environ,
            capture_output=True,
        )
        environ = DotEnv(None, stream=StringIO(process.stdout.decode("utf-8")), override=True)
        environ.set_as_environment_variables()

    def cleanup(self) -> None:
        shutil.rmtree(self.output_folder)

    @classmethod
    def prepare_build_environment(cls) -> None:
        self = cls()
        self.detect_profile()
        self.install()
        self.source()
        self.cleanup()
