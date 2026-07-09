# AMD leaderboard watch

A no-build Preact single page app for monitoring the AMD Developer Hackathon Act II leaderboard.

It polls:

`https://lablab.ai/api/v4/amd-developer-hackathon-act-ii/amd-leaderboard`

The app highlights the `yassai` submission and the `Solo Stack` team, filters to Track 1 by default, and ranks rows from the API rank where available. You can also force token-ascending ranking, which matches the Track 1 scoring rule from the participant guide: submissions that pass the accuracy gate are ranked by fewer recorded tokens.

## Run

No build step is required.

```sh
cd leaderboard-watch
python3 -m http.server 8080
```

Then open <http://localhost:8080>.

## Notes on the API shape

The endpoint response is an object with `available` and `tracks`. Each track contains `track`, `name`, `rankBy`, `entries`, and `failedEntries`. Track 1 is `General-Purpose AI Agent` and declares `rankBy: "tokens-asc"`, so lower token counts rank higher among evaluated submissions. The app still scans nested arrays and normalises common field names to be resilient to small response changes.
