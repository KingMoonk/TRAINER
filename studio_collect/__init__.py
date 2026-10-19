"""Passive training-data collection: open datasets by topic, question-and-answer sources, and the background service that runs them.

Records are cleaned (personal data, quality, duplicates, held-out text) and handed to the app as shards under data/collect/shards/;
`studio_collect.register` adds them to the workspace's text sets, where the trainer reads them. See issue #447.
"""
