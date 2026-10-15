"""Small verified public-source parsers; fetching is explicit and provenance-gated."""

from __future__ import annotations

import csv
import io
import itertools
import math
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from .artifacts import RawArtifactStore
from .models import RawArtifactManifest, SourceTerms
from .resampling import IntervalSample


def fetch_public(
    url: str,
    *,
    store: RawArtifactStore,
    source_id: str,
    product_id: str,
    terms: SourceTerms,
    max_bytes: int = 20_000_000,
    timeout_seconds: float = 30.0,
) -> RawArtifactManifest:
    """Bounded HTTPS download; reviewed terms are mandatory, no embedded credentials.

    Authenticated ERCOT access is deliberately not implemented here. Use a reviewed
    private downloader and feed exact bytes to RawArtifactStore instead.
    """
    parts = urlparse(url)
    allowed = {"reports-public.ieso.ca", "www.ncei.noaa.gov"}
    if (
        parts.scheme != "https"
        or parts.hostname not in allowed
        or parts.username
        or parts.password
        or parts.fragment
    ):
        raise ValueError("only credential-free HTTPS IESO/NOAA URLs are supported")
    if any(s in parts.query.lower() for s in ("key=", "token=", "secret=", "password=")):
        raise ValueError("credentials must not be embedded in URLs")
    if max_bytes <= 0 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("positive finite download limits required")
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        if urlparse(response.geturl()).hostname not in allowed:
            raise ValueError("unexpected redirect outside approved data hosts")
        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError("response exceeds configured byte limit")
        return store.put(
            payload,
            source_id=source_id,
            product_id=product_id,
            canonical_url=url,
            retrieved_at_utc=datetime.now(timezone.utc),
            content_type=response.headers.get_content_type(),
            parser_version="1.0.0",
            terms=terms,
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )


def parse_ieso_zonal_xml(payload: bytes) -> list[IntervalSample]:
    """Renewed-market RealTimeOntarioZonalPrice r2, LmpCap in CAD/MWh.

    IESO delivery hours are hour-ending in fixed Eastern Standard Time (UTC-5),
    not daylight-saving local Toronto time. Interval numbers are 1..12.
    Different report schemas/regimes are rejected instead of mixed with HOEP.
    """
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError("DTD/entity declarations are not allowed")
    root = ET.fromstring(payload)
    if root.attrib.get("docID") != "RealtimeOntarioZonalPrice":
        raise ValueError("unsupported IESO report regime")
    ns = {"i": "http://www.ieso.ca/schema"}
    date = root.findtext("i:DocBody/i:DeliveryDate", namespaces=ns)
    hour = root.findtext("i:DocBody/i:DeliveryHour", namespaces=ns)
    if date is None or hour is None or not 1 <= int(hour) <= 24:
        raise ValueError("invalid delivery date/hour")
    beginning = datetime.fromisoformat(date).replace(
        tzinfo=timezone(timedelta(hours=-5))
    ) + timedelta(hours=int(hour) - 1)
    result = []
    seen = set()
    for element in root.findall("i:DocBody/i:ZonalPrice", ns):
        interval = int(element.findtext("i:Interval", namespaces=ns) or "0")
        if not 1 <= interval <= 12 or interval in seen:
            raise ValueError("duplicate or invalid IESO interval")
        seen.add(interval)
        price = float(element.findtext("i:LmpCap", namespaces=ns) or "nan")
        start = (beginning + timedelta(minutes=(interval - 1) * 5)).astimezone(timezone.utc)
        result.append(IntervalSample(start, start + timedelta(minutes=5), price))
    if not result:
        raise ValueError("report contains no prices")
    return sorted(result, key=lambda r: r.start_utc)


def parse_noaa_temperatures(payload: bytes, *, station: str) -> list[IntervalSample]:
    """ISD global-hourly TMP tenths Celsius; trusted QC only, no future backfill.

    Each record is an instantaneous observation, NOT an hour-average. Resample
    using STATE semantics and an explicit max_state_age to avoid temporal leakage.
    """
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if not {"STATION", "DATE", "TMP"} <= set(reader.fieldnames or []):
        raise ValueError("NOAA station/date/TMP columns missing")
    readings: dict[datetime, float] = {}
    for row in reader:
        if row["STATION"] != station:
            continue
        raw, quality = row["TMP"].split(",")
        if raw in {"+9999", "9999"} or quality not in {"0", "1", "4", "5"}:
            continue
        temp = int(raw) / 10
        stamp = datetime.fromisoformat(row["DATE"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)  # ISD DATE explicitly UTC
        stamp = stamp.astimezone(timezone.utc)
        if stamp in readings and readings[stamp] != temp:
            raise ValueError("conflicting NOAA observations at same timestamp")
        readings[stamp] = temp
    if not readings:
        raise ValueError("no trustworthy station observations")
    return [IntervalSample(t, t + timedelta(seconds=1), v) for t, v in sorted(readings.items())]


def parse_google_task_usage(
    payload: bytes, *, trace_origin_utc: datetime, compute_units_per_cpu_hour: float
) -> list[IntervalSample]:
    """Google 2011 v2.1 task_usage: microsecond start/end, normalized CPU column 6.

    Mapping CPU rate to abstract useful compute is explicitly synthetic and caller-
    calibrated; it is not measured power or an inferred SLA. Overlapping distinct
    tasks are summed; duplicate/overlapping records for the same task are rejected.
    Pass decompressed CSV shards after retaining exact original bytes in raw storage.
    """
    from .models import QualityFlag

    if trace_origin_utc.tzinfo is None or trace_origin_utc.utcoffset() != timedelta(0):
        raise ValueError("trace origin must be aware UTC")
    if not math.isfinite(compute_units_per_cpu_hour) or compute_units_per_cpu_hour <= 0:
        raise ValueError("explicit positive finite compute calibration required")
    events: dict[datetime, float] = {}
    tasks: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for number, row in enumerate(csv.reader(io.StringIO(payload.decode("utf-8"))), start=1):
        if len(row) != 20:
            raise ValueError(f"Google v2.1 row {number} must contain 20 columns")
        start, end = int(row[0]), int(row[1])
        cpu = float(row[5]) if row[5] else math.nan
        if start < 0 or end <= start or not math.isfinite(cpu) or cpu < 0:
            raise ValueError(f"invalid or missing Google task usage row {number}")
        tasks.setdefault((row[2], row[3]), []).append((start, end))
        a, b = (
            trace_origin_utc + timedelta(microseconds=start),
            trace_origin_utc + timedelta(microseconds=end),
        )
        rate = cpu * compute_units_per_cpu_hour
        events[a] = events.get(a, 0.0) + rate
        events[b] = events.get(b, 0.0) - rate
    if not events:
        raise ValueError("empty Google trace")
    for intervals in tasks.values():
        intervals.sort()
        for previous, current in itertools.pairwise(intervals):
            if current[0] < previous[1]:
                raise ValueError("duplicate or overlapping records for the same Google task")
    times = sorted(events)
    rate = 0.0
    result = []
    mask = int(QualityFlag.DERIVED | QualityFlag.SYNTHETIC)
    for index, timestamp in enumerate(times[:-1]):
        rate += events[timestamp]
        ending = times[index + 1]
        total = max(0.0, rate) * (ending - timestamp).total_seconds() / 3600
        result.append(IntervalSample(timestamp, ending, total, mask))
    return result
