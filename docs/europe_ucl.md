# Champions League 2026/27: who wins? (indicative)

**Answer: nobody stands out.** Arsenal, Manchester City, Barcelona, Real Madrid and Paris SG are each at about 13 % (Bayern ≈ 10 %, Liverpool ≈ 6 %, Inter ≈ 3 %). Differences between the first five are far smaller than the simulation noise and the model's uncertainty.

## Data
- **Fixtures and results**: the English Wikipedia page of the 2026/27 league phase (144 matches, 36 clubs). On 27 Oct 2026 only **29 matches** were recorded there, although more had been played (the page lags): unrecorded results are simulated, which loses information but adds no bias. Wikipedia text is CC BY-SA; only scores and dates are used.
- **Squad value and domestic results**: the CC0 Transfermarkt dataset (valuations to June 2026), plus the football-data.co.uk tables already used for the leagues (current season).
- **Three clubs are absent from the dataset** (AEK Athens, Sabah, Slovan Bratislava): assumed to have the lowest squad value of the field and a median domestic form. Non-Big-5 clubs use their 2025/26 domestic form (no 2026/27 data collected).

## Model (`src/olpred/europe.py`)
Multinomial logistic regression on two differences: log squad value and domestic points per match (last 20 league games), fitted on 3,085 Champions / Europa / Conference League matches 2012–2025. Walk-forward test on 943 matches of 2023–2025:

| Model | Log loss |
|---|---|
| Class frequencies | 1.0351 |
| Squad value only | 0.9840 (vs frequencies: −0.051, CI [−0.078, −0.022]) |
| Domestic form only | 1.0368 |
| **Value + domestic form** | 0.9834 (vs value only: −0.0005, CI [−0.0048, +0.0037]) |

So squad value carries the signal; **domestic form adds nothing measurable** once value is known. It is kept because the question asked to combine them; its weight is small.

## Simulation (`src/olpred/europe_sim.py`, 20,000 runs)
Remaining league-phase matches (outcome from the model, score drawn from past European matches with the same outcome), table with points / goal difference / goals scored, play-offs 9–24, round of 16 to final (two legs; extra time and penalties near a coin flip; final on neutral ground). Each club's strength gets a random shock (sd 0.15 in log value).

## Limits
Validated on single matches, **not on whole tournaments**. Bracket: structure follows 2024/25 but pairings inside a tier are random. Strength shock size is an assumption (title odds of the top five stay between 12 % and 14 % at sd 0 and 0.3). No injuries, line-ups or draw luck beyond that. Not betting advice.
