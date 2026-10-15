import re

# Label lines like ".. _topics-auth-customizing:"
LABEL_RE = re.compile(r"^[ \t]*\.\. _[\w -]+:[ \t]*$\n?", re.MULTILINE)

# Roles like :class:`~a.b.C`, :ref:`text <label>`, :py:meth:`x`
ROLE_RE = re.compile(r":(?:[\w+-]+:)*([\w+-]+):`([^`]+)`")


def _role_sub(m: re.Match) -> str:
    role, content = m.group(1), m.group(2)
    # "link text <target>" -> "link text"
    if role == "ticket":
        return f"ticket {content}"
    if content.endswith(">") and "<" in content:
        content = content.rsplit("<", 1)[0].strip()
    # "~a.b.C" -> "C"
    if content.startswith("~"):
        content = content[1:].split(".")[-1]
    return content.lstrip("!.")


def clean_chunk(text: str) -> str:
    text = LABEL_RE.sub("", text)
    text = ROLE_RE.sub(_role_sub, text)
    return text.strip("\n")
