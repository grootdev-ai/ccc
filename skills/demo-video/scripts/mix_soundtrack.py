#!/usr/bin/env python
"""Lay the voiced lines on the timeline, set a music bed under them, and master the soundtrack.

    python mix_soundtrack.py placement.json --clips clips [--music bed.m4a] [--duck 8]
                             [--length 629.6 | --film film.mp4] [--out soundtrack]
                             [--mux film-narrated.mp4]

placement.json is a list of {"id": "A1", "at": 12.4}: each voiced line (clips/<id>.wav from
voice_lines.py) and the second it starts. A line that runs into the next line's start is a
defect in the cut, not something to hide in the mix, so the script refuses it.

Writes <out>/narration.wav (the voice stem), <out>/soundtrack.wav and <out>/soundtrack.m4a
(AAC 192k, 48 kHz stereo, two-pass loudnorm to -16 LUFS, -1.5 dBTP). With --mux, puts it on the
film by stream copy. Needs ffmpeg and ffprobe on PATH, plus numpy and soundfile.

The numbers (the -23 LUFS voice stem, the 8 dB dip, the 0.3 s ramps, the -16 LUFS master) and
why they are what they are: references/soundtrack.md.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

import numpy as np
import soundfile as sf

SR = 48000
VOICE_LUFS = -23.0      # the narration stem before the master
PAD, RAMP = 0.3, 0.3    # the dip window pads each line this much each side; the ramps are this long
FADE_OUT = 3.0          # the bed fades over the film's last seconds
MASTER = "I=-16:TP=-1.5:LRA=11"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{' '.join(cmd[:4])} ... failed:\n{r.stderr[-1500:]}")
    return r


def decode(path):
    """Any audio file ffmpeg reads -> mono float32 at 48 kHz."""
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                        "-f", "f32le", "-"], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"ffmpeg could not decode {path}:\n{r.stderr.decode()[-800:]}")
    return np.frombuffer(r.stdout, np.float32).copy()


def lufs(path):
    err = run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "ebur128", "-f", "null", "-"]).stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", err)[-1])


def duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]).stdout
    return float(out.strip())


def box(x, k):
    """A k-sample moving average by running sum: np.convolve's numbers, in seconds rather than minutes."""
    if k <= 1:
        return x
    c = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    n, h = len(x), (k - 1) // 2
    m = np.arange(n) + h
    return ((c[np.minimum(m, n - 1) + 1] - c[np.maximum(0, m - k + 1)]) / k).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("placement", help="JSON list of {id, at}")
    ap.add_argument("--clips", default="clips")
    ap.add_argument("--music", help="the music bed (any format ffmpeg reads); looped or cut to length")
    ap.add_argument("--duck", type=float, default=8.0, help="dB the bed dips under speech (default 8)")
    ap.add_argument("--length", type=float,
                    help="soundtrack length in seconds (default: the film's, else the last line's end + 1 s)")
    ap.add_argument("--film", help="the picture; sets --length from it, and is what --mux writes onto")
    ap.add_argument("--out", default="soundtrack")
    ap.add_argument("--mux", help="write the film with this soundtrack to this path (video stream-copied)")
    args = ap.parse_args()

    place = sorted(json.loads(pathlib.Path(args.placement).read_text()), key=lambda p: p["at"])
    clips = pathlib.Path(args.clips)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. The voice stem: each line at its second. A line that runs into the next is refused.
    audio = {p["id"]: decode(clips / f"{p['id']}.wav") for p in place}
    last_end = max((p["at"] + len(audio[p["id"]]) / SR for p in place), default=0.0)
    length = args.length or (duration(args.film) if args.film else last_end + 1.0)
    n = int(round(length * SR))
    voice = np.zeros(n, np.float32)
    mask = np.zeros(n, np.float32)
    for i, p in enumerate(place):
        a = audio[p["id"]]
        at = int(p["at"] * SR)
        end = at + len(a)
        if i + 1 < len(place) and end / SR > place[i + 1]["at"]:
            sys.exit(f"{p['id']} runs {end / SR - place[i + 1]['at']:.2f}s into {place[i + 1]['id']}: "
                     "fix the cut, not the mix")
        if end > n:
            sys.exit(f"{p['id']} ends at {end / SR:.1f}s, past the {length:.1f}s soundtrack")
        voice[at:end] += a
        mask[max(0, at - int(PAD * SR)):min(n, end + int(PAD * SR))] = 1
    sf.write(out / "narration.wav", voice, SR)
    gain = 10 ** ((VOICE_LUFS - lufs(out / "narration.wav")) / 20)
    print(f"{len(place)} lines, last ends {last_end:.1f}s, soundtrack {length:.1f}s, "
          f"voice gain {20 * np.log10(gain):+.1f} dB")

    # 2. The bed: the music at its own level where nobody speaks, dipped under the lines.
    mix = voice * gain
    if args.music:
        music = decode(args.music)
        if len(music) < n:
            reps = int(np.ceil(n / len(music)))
            print(f"music is {len(music) / SR:.0f}s, film {length:.0f}s: looping it {reps}x "
                  "(choose a longer track if the join is audible)")
            music = np.tile(music, reps)
        music = music[:n].copy()
        fade = min(n, int(FADE_OUT * SR))
        music[n - fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        duck = box(mask, int(RAMP * SR)).clip(0, 1)
        bed = music * (1 - duck * (1 - 10 ** (-args.duck / 20)))
        mix = mix + bed
        print(f"bed dips {args.duck:g} dB under speech; speech covers {100 * mask.mean():.0f}% of the film")
    sf.write(out / "soundtrack.wav", mix, SR)

    # 3. Master: measure, then apply the measured values (two-pass loudnorm). Stereo first, so the
    #    measurement is of what ships: a mono measure followed by an upmix lands about 3 dB hot.
    probe = run(["ffmpeg", "-hide_banner", "-i", str(out / "soundtrack.wav"), "-af",
                 f"aformat=channel_layouts=stereo,loudnorm={MASTER}:print_format=json", "-f", "null", "-"]).stderr
    st = json.loads(probe[probe.rindex("{"):probe.rindex("}") + 1])
    second = (f"aformat=channel_layouts=stereo,loudnorm={MASTER}:measured_I={st['input_i']}:"
              f"measured_TP={st['input_tp']}:measured_LRA={st['input_lra']}:"
              f"measured_thresh={st['input_thresh']}:offset={st['target_offset']}:linear=true")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / "soundtrack.wav"), "-af", second,
         "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", "2", str(out / "soundtrack.m4a")])
    print(f"master: {float(st['input_i']):.1f} LUFS -> {lufs(out / 'soundtrack.m4a'):.1f} LUFS, "
          f"{out / 'soundtrack.m4a'}")

    # 4. Put it on the picture.
    if args.mux:
        if not args.film:
            sys.exit("--mux needs --film")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.film, "-i", str(out / "soundtrack.m4a"),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy", "-shortest",
             "-movflags", "+faststart", args.mux])
        print(f"muxed: {args.mux}")


if __name__ == "__main__":
    main()
