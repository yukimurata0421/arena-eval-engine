import argparse
import io
import re
import tokenize
from pathlib import Path

from deep_translator import GoogleTranslator

JP_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")
STR_RE = re.compile(r"(?s)^([rRbBuUfF]*)(\'\'\'|\"\"\"|'|\")(.*)(\2)$")

PH_PATTERNS = [
    re.compile(r"\{[^{}]*\}"),
    re.compile(r"%[-+#0-9.]*[sdif]"),
    re.compile(r"`[^`]+`"),
    re.compile(r"\b[A-Z_][A-Z0-9_]{2,}\b"),
    re.compile(r"--[A-Za-z0-9-]+"),
    re.compile(r"\[[A-Z/]+\]"),
    re.compile(r"\\n|\\t|\\r"),
]


def protect(text: str):
    placeholders = []
    out = text
    for pat in PH_PATTERNS:
        while True:
            m = pat.search(out)
            if not m:
                break
            token = f"__PH{len(placeholders)}__"
            placeholders.append(m.group(0))
            out = out[:m.start()] + token + out[m.end():]
    return out, placeholders


def restore(text: str, placeholders):
    out = text
    for i, orig in enumerate(placeholders):
        out = out.replace(f"__PH{i}__", orig)
    return out


class Translator:
    def __init__(self) -> None:
        self._tr = GoogleTranslator(source="ja", target="en")
        self._cache: dict[str, str] = {}

    def translate(self, text: str) -> str:
        if not JP_RE.search(text):
            return text
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        masked, ph = protect(text)
        try:
            out = self._tr.translate(masked)
        except Exception:
            out = masked
        out = restore(out, ph)
        out = out.replace("\u3000", " ")
        out = re.sub(r"\s+", " ", out).strip()
        self._cache[text] = out
        return out


def translate_string_token(tok: str, tr: Translator) -> str:
    m = STR_RE.match(tok)
    if not m:
        return tok
    prefix, quote, body, _ = m.groups()
    if not JP_RE.search(body):
        return tok

    new_body = tr.translate(body)

    if len(quote) == 1:
        q = quote
        alt = '"' if q == "'" else "'"

        if q in new_body:
            if alt not in new_body:
                q = alt
            else:
                if "r" in prefix.lower():
                    return tok
                new_body = new_body.replace(q, "\\" + q)
        return f"{prefix}{q}{new_body}{q}"

    return f"{prefix}{quote}{new_body}{quote}"


def process_py(path: Path, tr: Translator) -> bool:
    src = path.read_text(encoding="utf-8")
    out_tokens = []
    changed = False

    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        ttype, tstr, start, end, line = tok

        if ttype == tokenize.STRING and JP_RE.search(tstr):
            new = translate_string_token(tstr, tr)
            if new != tstr:
                changed = True
                tok = tokenize.TokenInfo(ttype, new, start, end, line)

        elif ttype == tokenize.COMMENT and JP_RE.search(tstr):
            m = re.match(r"^(#\s?)(.*)$", tstr)
            if m:
                prefix, body = m.groups()
            else:
                prefix, body = "#", tstr[1:]
            new_body = tr.translate(body)
            new = f"{prefix}{new_body}"
            if new != tstr:
                changed = True
                tok = tokenize.TokenInfo(ttype, new, start, end, line)

        out_tokens.append(tok)

    if not changed:
        return False

    path.write_text(tokenize.untokenize(out_tokens), encoding="utf-8", newline="\n")
    return True


def process_text(path: Path, tr: Translator) -> bool:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    out = []
    changed = False

    for line in lines:
        if not JP_RE.search(line):
            out.append(line)
            continue

        nl = "\n" if line.endswith("\n") else ""
        core = line[:-1] if nl else line
        leading = re.match(r"^\s*", core).group(0)
        body = core[len(leading):]
        trans = tr.translate(body)
        new_line = leading + trans + nl
        if new_line != line:
            changed = True
        out.append(new_line)

    if changed:
        path.write_text("".join(out), encoding="utf-8", newline="\n")
    return changed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--list", required=True)
    p.add_argument("--mode", choices=["py", "text"], required=True)
    args = p.parse_args()

    root = Path(args.root)
    list_path = root / args.list
    files = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    tr = Translator()
    changed = []

    for rel in files:
        path = root / rel
        if not path.exists():
            continue
        try:
            did = process_py(path, tr) if args.mode == "py" else process_text(path, tr)
            if did:
                changed.append(rel)
        except Exception as exc:
            print(f"[WARN] failed: {rel}: {exc}")

    print(f"mode={args.mode} files={len(files)} changed={len(changed)}")
    for rel in changed:
        print(rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
