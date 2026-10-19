"""HTML to text, with what the crawler needs from a page: title, links, JSON-LD blocks, robots meta tags. Standard library only."""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin

SKIP = {"script", "style", "noscript", "template", "svg", "iframe", "canvas", "head"}
CHROME = {"nav", "header", "footer", "aside", "form", "button", "select", "menu"}    # page furniture, not content
PARAGRAPH = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "section", "article", "main", "ul", "ol", "table"}    # a blank line around
LINE = {"br", "li", "tr", "dd", "dt", "hr"}                                                                                       # a line break
VOID = {"br", "hr", "img", "meta", "link", "input", "source", "wbr", "area", "base", "col", "embed", "param", "track"}


class Page(HTMLParser):
    """Feed it HTML; read .title, .text (the main content when the page marks one, else everything but the furniture), .links,
    .jsonld (parsed JSON-LD values) and .robots (the content of <meta name=robots|googlebot|ai ...>)."""

    def __init__(self, base=""):
        super().__init__(convert_charrefs=True)
        self.base, self.title, self.links, self.jsonld, self.robots = base, "", [], [], []
        self._stack, self._skip, self._chrome, self._main, self._pre = [], 0, 0, 0, 0
        self._all, self._content, self._in_title, self._ld, self._ld_buffer = [], [], False, False, []

    def _out(self, text):
        if self._skip or self._chrome:
            return
        self._all.append(text)
        if self._main:
            self._content.append(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta" and (a.get("name") or "").lower() in ("robots", "googlebot", "ai", "ai-training", "tdm-reservation"):
            self.robots.append(f"{(a.get('name') or '').lower()}={(a.get('content') or '').lower()}")
        if tag == "a" and a.get("href"):
            self.links.append(urldefrag(urljoin(self.base, a["href"]))[0])
        if tag == "script" and "ld+json" in (a.get("type") or "").lower():
            self._ld, self._ld_buffer = True, []
        if tag == "title":
            self._in_title = True
        if tag in VOID:
            if tag == "br":
                self._out("\n")
            return
        self._stack.append(tag)
        if tag in SKIP:
            self._skip += 1
        elif tag in CHROME:
            self._chrome += 1
        elif tag in ("main", "article"):
            self._main += 1
        elif tag == "pre":
            self._pre += 1
            self._out("\n```\n")
        if tag in PARAGRAPH and not self._pre:
            self._out("\n\n")
        elif tag in LINE and not self._pre:
            self._out("\n")

    def handle_endtag(self, tag):
        if tag == "script" and self._ld:
            self._ld = False
            try:
                self.jsonld.append(json.loads("".join(self._ld_buffer)))
            except ValueError:
                pass
        if tag == "title":
            self._in_title = False
        if tag not in self._stack:
            return
        while self._stack:       # close any tags left open inside this one
            top = self._stack.pop()
            if top in SKIP:
                self._skip -= 1
            elif top in CHROME:
                self._chrome -= 1
            elif top in ("main", "article"):
                self._main -= 1
            elif top == "pre":
                self._pre -= 1
                self._out("\n```\n")
            if top == tag:
                break
        if tag in PARAGRAPH:
            self._out("\n\n")

    def handle_data(self, data):
        if self._ld:
            self._ld_buffer.append(data)
        if self._in_title:
            self.title += data
        if self._skip:
            return
        self._out(data if self._pre else re.sub(r"\s+", " ", data))

    @property
    def text(self):
        chosen = self._content if sum(map(len, self._content)) > 400 else self._all
        return tidy("".join(chosen))


def tidy(text):
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def to_text(html, base=""):
    page = Page(base)
    page.feed(html)
    page.close()
    return page.text


def parse(html, base=""):
    page = Page(base)
    page.feed(html)
    page.close()
    return page


def qa_pairs(jsonld):
    """[(question, answer, answer_url, author)] from schema.org QAPage / Question blocks: the accepted answer, else the most upvoted."""
    found = []

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            kinds = node.get("@type")
            kinds = kinds if isinstance(kinds, list) else [kinds]
            if "Question" in kinds:
                found.append(node)
            for key in ("@graph", "mainEntity"):
                if key in node:
                    walk(node[key])
    walk(jsonld)
    out = []
    for q in found:
        answers = q.get("acceptedAnswer") or q.get("suggestedAnswer") or []
        answers = answers if isinstance(answers, list) else [answers]
        answers = [a for a in answers if isinstance(a, dict) and a.get("text")]
        if not answers or not (q.get("text") or q.get("name")):
            continue
        best = max(answers, key=lambda a: int(a.get("upvoteCount") or 0) if str(a.get("upvoteCount") or 0).lstrip("-").isdigit() else 0) if not q.get("acceptedAnswer") else answers[0]
        author = best.get("author")
        author = author.get("name") if isinstance(author, dict) else author
        name, body = str(q.get("name") or ""), str(q.get("text") or "")
        question = name if not body or body == name else f"{name}\n\n{body}".strip()
        out.append((to_text(question) or question, to_text(str(best["text"])) or str(best["text"]), best.get("url", ""), author or ""))
    return out
