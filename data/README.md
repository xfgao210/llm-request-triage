# BurstGPT data

Run `python scripts/prepare_burstgpt.py` from the repository root. The source is
`HPMLL/BurstGPT`, pinned to `d895a53bb7b8ec137d0d2fe203b335835a78c10a`.
The CSV SHA256 must be
`46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a`.

`raw/` contains the verified source and upstream license. `processed/` contains
selected windows, a manifest, and a license copy. Both directories are ignored;
no third-party request dataset is redistributed in this change.

The parser retains ChatGPT API logs with finite nonnegative timestamps and
positive integer input/output lengths. Malformed/failed rows are counted and
excluded. Timestamps are source-relative seconds; rows are sorted by timestamp,
then their zero-based source row number. Equal timestamps stay together at the
60%/80% chronological split boundaries.

Defaults select 32 train, 8 validation, and 16 test windows of 256 requests.
Windows never reuse a source row. They are evenly spaced count-based samples,
not full-trace replay; timestamp ties may appear in different windows of the
same split. The output-length estimate uses selected training rows only.

Replay aligns the first arrival to zero. Native timing keeps all source gaps;
optional rate scaling multiplies every gap by one explicit factor. Deadlines
are `aligned arrival + configured completion SLO`. All arrivals lie strictly
inside the measurement horizon; the simulator can then drain outstanding work.

The source is [BurstGPT](https://github.com/HPMLL/BurstGPT), licensed under
[CC-BY-4.0](https://github.com/HPMLL/BurstGPT/blob/d895a53bb7b8ec137d0d2fe203b335835a78c10a/LICENSE).
Derived windows filter/subsample rows and replay may rescale timestamps. They
contain token metadata, not prompt text. Provenance and changes are recorded in
the manifest. Paper: https://doi.org/10.1145/3711896.3737413.
