"""Shared lightweight test helpers."""

from __future__ import annotations


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _Context:
    def __init__(self, streamlit):
        self.streamlit = streamlit

    def __enter__(self):
        return self.streamlit

    def __exit__(self, exc_type, exc, tb):
        return False
