from abc import ABC, abstractmethod

from graph.edge import Edge


class LanguagePlugin(ABC):
    extensions: list[str] = []

    @abstractmethod
    def extract_edges(self, file_path: str, source: str) -> list[Edge]:
        """Extract structural edges from a source file. Returns [] on parse error."""


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, LanguagePlugin] = {}

    def register(self, plugin: LanguagePlugin) -> None:
        for ext in plugin.extensions:
            self._plugins[ext] = plugin

    def get(self, ext: str) -> LanguagePlugin | None:
        return self._plugins.get(ext)


default_registry = PluginRegistry()
