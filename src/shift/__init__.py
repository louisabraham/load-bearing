"""Abrupt shifts in GitHub comment vocabulary.

A deliberately small pipeline: one uniform data stream, whitespace words counted
once per document, and a changepoint scan on the word-frequency vector.
"""
