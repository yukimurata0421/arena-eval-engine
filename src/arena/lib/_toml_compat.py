"""Shared TOML parsing utilities (avoids circular imports between paths and runtime_config)."""

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        tomllib = None


def parse_settings_fallback(text: str) -> dict:
    """Minimal INI-style parser for settings.toml when tomllib is unavailable."""
    settings: dict = {}
    section = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            settings.setdefault(section, {})
            continue
        if "=" in line and section:
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            settings.setdefault(section, {})[key] = val
    return settings
