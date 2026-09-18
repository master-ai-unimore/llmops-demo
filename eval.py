#!/usr/bin/env python3
"""Eval harness della demo LLMOps: confronta versioni di prompt sul golden set.

Solo stdlib. Provider: Ollama locale (http://localhost:11434).

    python3 eval.py                     # tutti i prompt, tutti i 30 ticket
    python3 eval.py --limite 10         # prova veloce
    python3 eval.py --modello qwen2.5:7b
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

QUI = Path(__file__).parent
CATEGORIE = ["prezzo", "qualita_rete", "concorrenza", "servizio_clienti", "altro"]
OLLAMA_URL = "http://localhost:11434/api/chat"


def chiedi(modello: str, prompt: str) -> tuple[str, float]:
    """Una chiamata al modello locale; ritorna (risposta, latenza in secondi)."""
    corpo = json.dumps({
        "model": modello,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }).encode()
    inizio = time.perf_counter()
    req = urllib.request.Request(OLLAMA_URL, data=corpo,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        risposta = json.loads(r.read())["message"]["content"]
    return risposta, time.perf_counter() - inizio


def estrai_categoria(risposta: str) -> str | None:
    """La risposta è una categoria? Tollera maiuscole e punteggiatura di contorno,
    NON tollera le chiacchiere: se il testo non è riconducibile a una sola
    categoria, è un errore di formato (e in produzione, un parser rotto)."""
    pulita = risposta.strip().lower().strip(".:\"' \n")
    pulita = pulita.replace("qualità rete", "qualita_rete").replace("servizio clienti", "servizio_clienti")
    if pulita in CATEGORIE:
        return pulita
    trovate = {c for c in CATEGORIE if re.search(rf"\b{c}\b", pulita)}
    return trovate.pop() if len(trovate) == 1 else None


def valuta_prompt(nome: str, sagoma: str, golden: list[dict], modello: str) -> dict:
    corrette = non_parseabili = 0
    latenze = []
    for caso in golden:
        risposta, latenza = chiedi(modello, sagoma.replace("{ticket}", caso["testo"]))
        latenze.append(latenza)
        categoria = estrai_categoria(risposta)
        if categoria is None:
            non_parseabili += 1
        elif categoria == caso["categoria"]:
            corrette += 1
        print(f"  [{nome}] ticket {caso['id']:>2}: attesa={caso['categoria']:<17}"
              f" -> {categoria or 'FUORI FORMATO'}", file=sys.stderr)
    n = len(golden)
    return {
        "prompt": nome,
        "accuracy": corrette / n,
        "non_parseabili": non_parseabili,
        "latenza_media_s": sum(latenze) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Confronta i prompt sul golden set")
    ap.add_argument("--modello", default="qwen2.5:7b")
    ap.add_argument("--limite", type=int, help="usa solo i primi N ticket")
    args = ap.parse_args()

    golden = [json.loads(r) for r in (QUI / "golden_set.jsonl").read_text().splitlines() if r]
    if args.limite:
        golden = golden[: args.limite]

    try:
        chiedi(args.modello, "ping")
    except (urllib.error.URLError, OSError):
        sys.exit("Ollama non raggiungibile su localhost:11434 — avvialo con "
                 f"'ollama serve' e scarica il modello con 'ollama pull {args.modello}'")

    risultati = []
    for percorso in sorted(QUI.glob("prompt_v*.md")):
        print(f"\n=== {percorso.name} ===", file=sys.stderr)
        risultati.append(valuta_prompt(percorso.stem, percorso.read_text(), golden, args.modello))

    print(f"\nGolden set: {len(golden)} ticket · modello: {args.modello}\n")
    print(f"{'prompt':<12} {'accuracy':>9} {'fuori formato':>15} {'latenza media':>14}")
    for r in risultati:
        print(f"{r['prompt']:<12} {r['accuracy']:>8.0%} {r['non_parseabili']:>15} "
              f"{r['latenza_media_s']:>13.2f}s")


if __name__ == "__main__":
    main()
