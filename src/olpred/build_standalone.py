"""Bundle the website (HTML + CSS + fonts + JS + data) into ONE file you can open by double-click or send by mail.
Run:  python -m olpred.build_standalone   ->  site/ol-lab-standalone.html"""
import base64
import re
from pathlib import Path

SITE = Path("site")


def main() -> None:
    html = (SITE / "index.html").read_text()
    css, js, data = (SITE / "style.css").read_text(), (SITE / "app.js").read_text(), (SITE / "data.js").read_text()

    def inline_font(m: re.Match) -> str:
        raw = (SITE / m.group(1)).read_bytes()
        return f"url(data:font/woff2;base64,{base64.b64encode(raw).decode()}) format(\"woff2\")"
    css = re.sub(r"url\((fonts/[^)]+\.woff2)\) format\(\"woff2\"\)", inline_font, css)
    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}\n</style>")
    html = html.replace('<script src="data.js"></script>', f"<script>\n{data}\n</script>")
    html = html.replace('<script src="app.js"></script>', f"<script>\n{js}\n</script>")
    out = SITE / "ol-lab-standalone.html"
    out.write_text(html)
    print(out, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
