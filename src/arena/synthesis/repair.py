from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Repairs are intentionally conservative:
# - never mutate source files on disk
# - apply only low-risk syntactic transformations

_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)
_DOUBLED_QUOTE_TOKEN_RE = re.compile(r'^(?P<indent>\s*)""(?P<body>.*)""(?P<trail>\s*,?\s*)$')
_HALF_DOUBLED_QUOTE_TOKEN_RE = re.compile(r'^(?P<indent>\s*)""(?P<body>.*)"(?P<trail>\s*,?\s*)$')
_KEY_VALUE_SNIPPET_RE = re.compile(r'^[A-Za-z0-9_.-]+":\s')
_KEY_VALUE_DOUBLE_WRAPPED_RE = re.compile(
    r'^(?P<prefix>\s*"[^"\n\r]+"\s*:\s*)""(?P<body>.*)""(?P<suffix>\s*(?:[,}\]])?\s*)$'
)
_STRING_TOKEN_INNER_QUOTE_RE = re.compile(
    r'^(?P<indent>\s*)"(?P<prefix>[^"\n\r]*)"(?P<body>[^"\n\r]+)""(?P<trail>\s*,?\s*)$'
)

_NONE_LITERAL_RE = re.compile(r'(?P<prefix>[:\[,]\s*)None(?P<suffix>\s*[,}\]])')
_TRUE_LITERAL_RE = re.compile(r'(?P<prefix>[:\[,]\s*)True(?P<suffix>\s*[,}\]])')
_FALSE_LITERAL_RE = re.compile(r'(?P<prefix>[:\[,]\s*)False(?P<suffix>\s*[,}\]])')


@dataclass(frozen=True, slots=True)
class RepairResult:
    text: str
    applied_rules: tuple[str, ...]


def _strip_code_fence_wrapper(text: str) -> tuple[str, int]:
    matches = _CODE_FENCE_RE.findall(text)
    if not matches:
        return text, 0
    best = max(matches, key=len)
    if not best.strip():
        return text, 0
    return best.strip(), 1


def _extract_json_window(text: str) -> tuple[str, int]:
    start_candidates = [idx for idx in (text.find("["), text.find("{")) if idx >= 0]
    if not start_candidates:
        return text, 0

    start = min(start_candidates)
    end = max(text.rfind("]"), text.rfind("}"))
    if end <= start:
        return text, 0

    candidate = text[start : end + 1]
    if candidate == text:
        return text, 0
    if not candidate.strip():
        return text, 0
    return candidate, 1


def _replace_python_literals(text: str) -> tuple[str, int]:
    total = 0
    working = text
    working, c1 = _NONE_LITERAL_RE.subn(r"\g<prefix>null\g<suffix>", working)
    total += c1
    working, c2 = _TRUE_LITERAL_RE.subn(r"\g<prefix>true\g<suffix>", working)
    total += c2
    working, c3 = _FALSE_LITERAL_RE.subn(r"\g<prefix>false\g<suffix>", working)
    total += c3
    return working, total


def _repair_doubled_quote_tokens(text: str) -> tuple[str, int]:
    changed = 0
    repaired_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        newline = ""
        core = line
        if line.endswith("\r\n"):
            newline = "\r\n"
            core = line[:-2]
        elif line.endswith("\n"):
            newline = "\n"
            core = line[:-1]

        matched = _DOUBLED_QUOTE_TOKEN_RE.match(core)
        if matched is None:
            matched = _HALF_DOUBLED_QUOTE_TOKEN_RE.match(core)
        if matched is None:
            repaired_lines.append(line)
            continue

        body = matched.group("body")
        if _KEY_VALUE_SNIPPET_RE.match(body):
            if not body.startswith('"'):
                body = f'"{body}'
            if ': "' in body and not body.endswith('"'):
                body = f'{body}"'
        repaired = f"{matched.group('indent')}{json.dumps(body, ensure_ascii=False)}{matched.group('trail')}"
        repaired_lines.append(f"{repaired}{newline}")
        changed += 1

    return "".join(repaired_lines), changed


def _repair_key_value_double_wrapped_values(text: str) -> tuple[str, int]:
    changed = 0
    repaired_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        newline = ""
        core = line
        if line.endswith("\r\n"):
            newline = "\r\n"
            core = line[:-2]
        elif line.endswith("\n"):
            newline = "\n"
            core = line[:-1]

        matched = _KEY_VALUE_DOUBLE_WRAPPED_RE.match(core)
        if matched is None:
            repaired_lines.append(line)
            continue

        body = matched.group("body")
        # Join malformed quoted fragments such as: "line1"\n"line2"
        body = body.replace('"\\r\\n"', "\r\n")
        body = body.replace('"\\n"', "\n")
        body = body.replace('"\\t"', "\t")
        repaired = f"{matched.group('prefix')}{json.dumps(body, ensure_ascii=False)}{matched.group('suffix')}"
        repaired_lines.append(f"{repaired}{newline}")
        changed += 1

    return "".join(repaired_lines), changed


def _repair_string_tokens_with_inner_quote(text: str) -> tuple[str, int]:
    changed = 0
    repaired_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        newline = ""
        core = line
        if line.endswith("\r\n"):
            newline = "\r\n"
            core = line[:-2]
        elif line.endswith("\n"):
            newline = "\n"
            core = line[:-1]

        matched = _STRING_TOKEN_INNER_QUOTE_RE.match(core)
        if matched is None:
            repaired_lines.append(line)
            continue

        prefix = matched.group("prefix")
        body = matched.group("body")
        if not prefix.strip() or not body.strip() or "\\" in prefix or "\\" in body:
            repaired_lines.append(line)
            continue

        repaired = f'{matched.group("indent")}"{prefix}\\"{body}\\""{matched.group("trail")}'
        repaired_lines.append(f"{repaired}{newline}")
        changed += 1

    return "".join(repaired_lines), changed


def repair_json_text(text: str) -> RepairResult:
    working = text
    applied_rules: list[str] = []

    repaired, changed = _strip_code_fence_wrapper(working)
    if changed > 0:
        working = repaired
        applied_rules.append("strip_code_fence_wrapper:1")

    repaired, changed = _extract_json_window(working)
    if changed > 0:
        working = repaired
        applied_rules.append("extract_json_window:1")

    repaired, changed = _replace_python_literals(working)
    if changed > 0:
        working = repaired
        applied_rules.append(f"python_literal_to_json_literal:{changed}")

    repaired, changed = _repair_key_value_double_wrapped_values(working)
    if changed > 0:
        working = repaired
        applied_rules.append(f"double_wrapped_key_value_string:{changed}")

    repaired, changed = _repair_string_tokens_with_inner_quote(working)
    if changed > 0:
        working = repaired
        applied_rules.append(f"string_token_inner_quote_escape:{changed}")

    repaired, changed = _repair_doubled_quote_tokens(working)
    if changed > 0:
        working = repaired
        applied_rules.append(f"doubled_quote_string_tokens:{changed}")

    return RepairResult(text=working, applied_rules=tuple(applied_rules))
