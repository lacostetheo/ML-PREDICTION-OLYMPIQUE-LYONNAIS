"""Parsers of the public calendars (openfootball, Wikipedia UEFA pages) and the repair of a published calendar."""
import pandas as pd

from olpred.data import fixtures, openfootball, uefa

OPENFOOTBALL = """= English Premier League 2026/27

▪ Matchday 1
  Fri Aug 21 2026
    20:00  Arsenal FC              v Coventry City FC         3-0 (2-0)
  Sat Aug 22
    12:30  Hull City AFC           v Manchester United FC     2-0 (2-0)
           Everton FC              v Crystal Palace FC        2-0 (1-0)
    17:30  Brighton & Hove Albion FC v Aston Villa FC           4-0 (4-0)

▪ Matchday 20
  Sat Jan 2
    15:00  Arsenal FC              v Hull City AFC
"""


def test_openfootball_parser_reads_names_scores_times_and_the_new_year():
    df = openfootball.parse(OPENFOOTBALL, "E0")
    assert list(df["home"]) == ["Arsenal", "Hull", "Everton", "Brighton", "Arsenal"]
    assert list(df["away"]) == ["Coventry", "Man United", "Crystal Palace", "Aston Villa", "Hull"]
    assert df.loc[0, "hg"] == 3 and df.loc[0, "ag"] == 0 and pd.isna(df.loc[4, "hg"])
    assert df.loc[2, "time"] == "12:30"            # a blank time means "same slot as the line above"
    assert df.loc[4, "date"] == pd.Timestamp("2027-01-02")


WIKI = """==Matches==
===Matchday 1===
{{#invoke:Football box|main
|date       = {{Start date|2026|9|8|df=y}}
|time       = 18:45&nbsp;{{small|(19:45 [[UTC+03:00|UTC+3]])}}
|team1      = [[AEK Athens F.C.|AEK Athens]] {{fbaicon|GRE}}
|score      = 1–0
|team2      = {{fbaicon|AUT}} [[LASK]]
}}
----
{{#invoke:Football box|main
|date       = {{Start date|2026|11|3|df=y}}
|time       = 21:00
|team1      = [[Club Brugge KV|Club Brugge]] {{fbaicon|BEL}}
|score      =
|team2      = {{fbaicon|ENG}} [[Aston Villa F.C.|Aston Villa]]
}}
"""


def test_uefa_parser_reads_played_and_unplayed_matches():
    df = uefa.parse(WIKI)
    assert list(df["home"]) == ["AEK Athens", "Club Brugge"] and list(df["away"]) == ["LASK", "Aston Villa"]
    assert df.loc[0, "hg"] == 1 and df.loc[0, "ag"] == 0 and df.loc[0, "time"] == "18:45"
    assert pd.isna(df.loc[1, "hg"]) and df.loc[1, "date"] == "2026-11-03"


def test_a_calendar_that_lists_a_pair_twice_gets_its_return_leg_restored():
    cal = pd.DataFrame({"matchday": [1, 2, 3], "date": pd.to_datetime(["2026-08-22", "2026-08-29", "2027-03-06"]),
                        "home": ["Rennes", "Paris SG", "Rennes"], "away": ["Paris SG", "Lille", "Paris SG"]})
    fixed = fixtures.repair_duplicates(cal, "F1")
    assert (fixed.loc[fixed["matchday"] == 3, ["home", "away"]].iloc[0] == ["Paris SG", "Rennes"]).all()
    assert (fixed.loc[fixed["matchday"] == 1, ["home", "away"]].iloc[0] == ["Rennes", "Paris SG"]).all()
