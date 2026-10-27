"""Bundle the website (HTML + CSS + JS + data) into ONE file you can open by double-click or send by mail.
Run:  python -m olpred.build_standalone   ->  site/standalone/ol-lab.html"""
from pathlib import Path

SITE = Path("site")


def main() -> None:
    html = (SITE / "index.html").read_text()
    css, js, data = (SITE / "style.css").read_text(), (SITE / "app.js").read_text(), (SITE / "data.js").read_text()
    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}\n</style>")
    html = html.replace('<script src="data.js"></script>', f"<script>\n{data}\n</script>")
    html = html.replace('<script src="app.js"></script>', f"<script>\n{js}\n</script>")
    out = SITE / "ol-lab-standalone.html"
    out.write_text(html)
    print(out, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
