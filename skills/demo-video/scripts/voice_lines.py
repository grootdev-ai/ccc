#!/usr/bin/env python
"""Voice every narration line with Kokoro, trim its silence, and read it back with Whisper.

    python voice_lines.py lines.json [--out clips] [--voice af_heart] [--speed 1.0]
                          [--only A1,A2] [--respell respell.json] [--check [small.en|medium]]
                          [--check-only]

lines.json is a list of {"id": "A1", "text": "..."}; a pitch-craft script's segments map onto it
one to one. Writes <out>/<id>.wav (24 kHz mono) and <out>/clips.json with each line's duration,
words per second, voice and speed. With --check, transcribes each clip with faster-whisper, scores
it against its text, writes <out>/clips-checked.json and prints the report (mean match, the count
at or above the threshold, and every miss with what was heard).

Run it with a virtualenv that has kokoro==0.9.4, soundfile and numpy; --check also needs
faster-whisper. Why each number is what it is: references/soundtrack.md.
"""
import argparse
import difflib
import json
import pathlib
import re
import sys
import warnings

# numpy and soundfile load where they are used, so --check-only runs in a venv that has only
# faster-whisper (the Kokoro venv and the Whisper venv are often separate).
SR = 24000
GAP = 0.22          # silence between the sentence chunks Kokoro returns for one line
TRIM_DB = -42.0     # speech starts where the signal first exceeds this
LEAD, TAIL = 0.04, 0.08   # keep this much silence before and after the speech

# Voice-only rewrites, applied before synthesis and never shown in the script. Add the film's own
# in a respell.json: [["\\bPEP\\b", "P.E.P."], ["\\bCTOS\\b", "C-toss"], ...]. A proper noun the
# model stresses wrongly takes Kokoro's phoneme override: "[Petaling Jaya](/pətˈɑːlɪŋ ʤˈɑːjə/)".
DEFAULT_RESPELL = []


def load_respell(path):
    rules = list(DEFAULT_RESPELL)
    if path:
        rules += [tuple(r) for r in json.loads(pathlib.Path(path).read_text())]
    return [(re.compile(p), r) for p, r in rules]


def spoken(text, rules):
    for pattern, repl in rules:
        text = pattern.sub(repl, text)
    return text


def trim(audio):
    import numpy as np
    loud = np.nonzero(np.abs(audio) > 10 ** (TRIM_DB / 20))[0]
    if len(loud) == 0:
        return audio
    a0 = max(0, loud[0] - int(LEAD * SR))
    a1 = min(len(audio), loud[-1] + int(TAIL * SR))
    return audio[a0:a1]


def synth(pipe, text, voice, speed):
    import numpy as np
    parts = []
    for _, _, audio in pipe(text, voice=voice, speed=speed):
        a = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
        if parts:
            parts.append(np.zeros(int(GAP * SR), np.float32))
        parts.append(trim(a.astype(np.float32)))
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def norm(text):
    text = text.lower().replace("-", " ")
    return re.sub(r"[^a-z0-9' ]", " ", text).split()


def check(out, meta, model_name, threshold):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    low = []
    for m in meta:
        segs, _ = model.transcribe(str(out / f"{m['id']}.wav"), beam_size=5, language="en")
        heard = " ".join(s.text.strip() for s in segs)
        m["heard"] = heard
        m["match"] = round(difflib.SequenceMatcher(None, norm(m["text"]), norm(heard)).ratio(), 3)
        if m["match"] < threshold:
            low.append(m)
    (out / "clips-checked.json").write_text(json.dumps(meta, indent=1))
    mean = sum(m["match"] for m in meta) / len(meta)
    print(f"\n{len(meta)} lines, mean match {mean:.3f}, {len(meta) - len(low)} at or above "
          f"{threshold}, {len(low)} below")
    for m in low:
        print(f"  {m['id']} {m['match']}\n    said : {m['text']}\n    heard: {m['heard']}")
    if low:
        print("\nA number heard as digits is the transcriber. A product name or a figure heard "
              "wrong is a defect: respell it and re-voice that line with --only.")
    return low


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lines", help="JSON list of {id, text}")
    ap.add_argument("--out", default="clips")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--only", help="comma-separated ids to (re)voice; the rest keep their clips")
    ap.add_argument("--respell", help="JSON list of [regex, replacement] applied before synthesis")
    ap.add_argument("--check", nargs="?", const="small.en",
                    help="read every clip back with faster-whisper (model name; default small.en)")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--check-only", action="store_true",
                    help="skip synthesis and only read the existing clips back (needs --check)")
    args = ap.parse_args()
    if args.check_only and not args.check:
        ap.error("--check-only needs --check")

    lines = json.loads(pathlib.Path(args.lines).read_text())
    rules = load_respell(args.respell)
    only = set(args.only.split(",")) if args.only else None
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "clips.json"
    meta = {m["id"]: m for m in json.loads(meta_path.read_text())} if meta_path.exists() else {}

    warnings.filterwarnings("ignore")
    pipe = None
    for line in lines:
        if args.check_only or (only and line["id"] not in only):
            continue
        if pipe is None:
            from kokoro import KPipeline
            pipe = KPipeline(lang_code=args.voice[0], repo_id="hexgrad/Kokoro-82M")
        import soundfile as sf
        audio = synth(pipe, spoken(line["text"], rules), args.voice, args.speed)
        sf.write(out / f"{line['id']}.wav", audio, SR)
        dur = len(audio) / SR
        wps = round(len(line["text"].split()) / dur, 2) if dur else None
        meta[line["id"]] = {"id": line["id"], "text": line["text"], "duration": round(dur, 2),
                            "wps": wps, "voice": args.voice, "speed": args.speed}
        print(f"{line['id']:6s} {dur:6.2f}s  {wps} wps", flush=True)

    ordered = [meta[l["id"]] for l in lines if l["id"] in meta]
    meta_path.write_text(json.dumps(ordered, indent=1))
    print(f"{len(ordered)} clips in {out}/, {sum(m['duration'] for m in ordered):.1f}s of speech")
    if args.check:
        check(out, ordered, args.check, args.threshold)


if __name__ == "__main__":
    main()
