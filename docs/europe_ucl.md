# Champions League and Europa League 2026/27: who wins? (indicative)

State of 27 October 2026 (`python -m olpred.europe && python -m olpred.europe_sim`, 20,000 simulated competitions each).

| | Pick | Chance | Close behind |
|---|---|---|---|
| **Champions League** | Arsenal | 13.9 % | Paris SG 13.6 %, Man City 13.1 %, Real Madrid 12.4 %, Barcelona 12.1 % |
| **Europa League** | Bournemouth | 15.2 % | Juventus 14.2 %, Benfica 11.2 %, Leverkusen 10.7 %; **Lyon 4.0 %** |

**Nobody stands out in either competition.** The leader's lead is smaller than the simulation noise and than the uncertainty on club strength: with no strength shock at all Juventus becomes the Europa League pick. The site therefore labels both rows "no clear favourite" or "clubs within a point".

## Data
- **Fixtures and results**: the English Wikipedia pages of the 2026/27 league phases (144 matches and 36 clubs each, `src/olpred/data/uefa.py`). Only scores and dates are used. The pages lag behind reality: on 27 October 2026 they had recorded **29 of the 54** Champions League matches played and **54 of 54** Europa League matches. A match whose date has passed but has no score is simulated, not shown as upcoming, and counted in the "not recorded yet" note.
- **Squad value and domestic results**: the CC0 Transfermarkt dataset (valuations to June 2026), plus the football-data.co.uk tables for the current-season form of clubs of the five big leagues (other clubs use their 2025/26 form).
- **Clubs missing from the dataset** get the lowest squad value of the field and a median domestic form: 3 in the Champions League (AEK Athens, Sabah, Slovan Bratislava), 8 in the Europa League (Ararat-Armenia, Celje, Ferencváros, Hapoel Be'er Sheva, Levski Sofia, OFI, Omonia, Torreense). The site marks them with †.

## Model (`src/olpred/europe.py`)
Multinomial logistic regression on two differences between the clubs: log squad value and domestic points per match (last 20 league matches), fitted on 3,085 Champions / Europa / Conference League matches 2012–2025 (one model for all three competitions). Walk-forward test on 943 matches of 2023–2025:

| Model | Log loss |
|---|---|
| Class frequencies | 1.0351 |
| Squad value only | 0.9836 (vs frequencies: −0.052, CI [−0.079, −0.022]) |
| Domestic form only | 1.0368 |
| **Value + domestic form** | 0.9831 (vs value only: −0.0005, CI [−0.0048, +0.0038]) |

Squad value carries the signal; **domestic form adds nothing measurable** once value is known. It is kept because the question asked to combine them; its weight is small.

## Simulation (`src/olpred/europe_sim.py`)
Remaining league-phase matches (outcome from the model, score drawn from past European matches with the same outcome), table with points / goal difference / goals scored, play-offs 9–24, round of 16 to final (two legs; extra time and penalties close to a coin flip; final on neutral ground). The forecast of every match still to play (shown on the site) uses the strength of the clubs today, without shock.

## Limits
Validated on single matches, **not on whole tournaments**. Bracket: structure follows 2024/25 but pairings inside a tier are random. The strength shock (sd 0.15 in log squad value) is an assumption: at sd 0 and 0.3 the Champions League top five stay between 12 % and 14.5 %. The Europa League has more clubs outside the data, so its numbers are less reliable than the Champions League's. No injuries, line-ups or draw luck. Not betting advice.
