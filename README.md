# Demo LLMOps — Classificazione dei motivi di disdetta (pillola 17)

```bash
git clone https://github.com/master-ai-unimore/llmops-demo
```

Caso ChurnGuard-adiacente: il CRM di TelcoNova riceve ticket di disdetta in testo
libero; vogliamo classificarli per **motivo** (prezzo, qualità rete, concorrenza,
servizio clienti, altro) per alimentare le analisi churn.

La demo mostra la disciplina MLOps applicata agli LLM:

| Concetto del corso | Qui diventa |
|---|---|
| Iperparametri versionati (params.yaml) | **Il prompt in git**: `prompt_v1.md` vs `prompt_v2.md` — il diff è leggibile |
| Test set | **Golden set**: 30 ticket etichettati a mano (`golden_set.jsonl`) |
| `dvc metrics diff` | La tabella di `eval.py`: accuracy, risposte fuori formato, latenza |
| Il gate | "il prompt nuovo non peggiora sul golden set" |

## Esecuzione (tutto locale, via Ollama)

```bash
ollama pull qwen2.5:7b        # una volta sola
python3 eval.py               # valuta entrambi i prompt sul golden set
python3 eval.py --limite 10   # prova veloce sui primi 10 ticket
```

Nessuna dipendenza Python (solo stdlib). Con `--modello` si cambia modello; il
punto della demo non è il modello, è **l'harness**: golden set + confronto tra
versioni del prompt + metriche operative (latenza; con un provider a pagamento
la colonna costo si popola dai token).

## Numeri di riferimento (verificati, qwen2.5:7b, 30 ticket)

| prompt | accuracy | fuori formato | latenza media |
|---|---|---|---|
| prompt_v1 | 10% | 22/30 | 12,0s |
| prompt_v2 | **83%** | **0** | **0,69s** |

Tre lezioni in una tabella: (1) il prompt È il modello — stessa rete, risultati
opposti; (2) il vincolo di formato vale più di tutto (22 risposte-chiacchiera =
22 errori di parsing in produzione); (3) la latenza 17× più bassa di v2 è anche
il costo 17× più basso con un provider a token. E l'83% non-perfetto è il quarto
insegnamento: il golden set intercetta i casi ambigui — discutere "che ticket
sbaglia? serve un prompt_v3 o serve una categoria in più?".

## Cosa mostrare in video

1. `git diff` tra i due prompt: v1 è la richiesta ingenua, v2 aggiunge ruolo,
   definizioni delle categorie, formato di output vincolato e 2 esempi.
2. `python3 eval.py`: la tabella finale — v2 vince su accuracy e soprattutto su
   **risposte fuori formato** (v1 "chiacchiera", v2 risponde con la categoria).
3. Il messaggio: *nessuna ground truth in produzione → il golden set curato a
   mano è il vostro test set; si mantiene come un giardino.*
