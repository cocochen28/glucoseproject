"""Exploratory data analysis for the OhioT1DM XML dataset.

This script reads all patient XML files in a directory, extracts only the
glucose_level, meal, bolus, basal, and finger_stick sections, and reports
per-patient and aggregate statistics. It also produces up to three plots:
  1) Glucose histogram (all patients combined)
  2) Glucose time series (first 1–2 days for one patient)
  3) Meal carb distribution (falls back to insulin dose distribution)

Usage:
	python analyze_data.py --data-dir ./data/raw/ohio_t1dm --days 2

Dependencies: numpy, pandas, matplotlib (plus the Python standard library).
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


DATE_FMT = "%d-%m-%Y %H:%M:%S"


@dataclass
class PatientData:
	patient_id: str
	glucose: pd.DataFrame
	meals: pd.DataFrame
	boluses: pd.DataFrame
	basals: pd.DataFrame
	finger_sticks: pd.DataFrame


def parse_timestamp(ts: Optional[str]) -> pd.Timestamp:
	"""Parse timestamp strings like '07-12-2021 01:17:00' with day-first format."""

	if ts is None:
		return pd.NaT
	return pd.to_datetime(ts, format=DATE_FMT, dayfirst=True, errors="coerce")


def parse_glucose(root: ET.Element) -> pd.DataFrame:
	section = root.find("glucose_level")
	records: List[Dict[str, object]] = []
	if section is None:
		return pd.DataFrame(columns=["ts", "value"])

	for event in section.findall("event"):
		ts = parse_timestamp(event.get("ts"))
		value = event.get("value")
		if ts is pd.NaT or value is None:
			continue
		records.append({"ts": ts, "value": float(value)})

	df = pd.DataFrame(records)
	if not df.empty:
		df.sort_values("ts", inplace=True)
		df.reset_index(drop=True, inplace=True)
	return df


def parse_meal(root: ET.Element) -> pd.DataFrame:
	section = root.find("meal")
	records: List[Dict[str, object]] = []
	if section is None:
		return pd.DataFrame(columns=["ts", "carbs", "type"])

	for event in section.findall("event"):
		ts = parse_timestamp(event.get("ts"))
		carbs = event.get("carbs")
		meal_type = event.get("type")
		if ts is pd.NaT or carbs is None:
			continue
		records.append({"ts": ts, "carbs": float(carbs), "type": meal_type})

	df = pd.DataFrame(records)
	if not df.empty:
		df.sort_values("ts", inplace=True)
		df.reset_index(drop=True, inplace=True)
	return df


def parse_bolus(root: ET.Element) -> pd.DataFrame:
	section = root.find("bolus")
	records: List[Dict[str, object]] = []
	if section is None:
		return pd.DataFrame(columns=["ts", "dose", "type"])

	for event in section.findall("event"):
		ts = parse_timestamp(event.get("ts_begin"))
		dose = event.get("dose")
		bolus_type = event.get("type")
		if ts is pd.NaT or dose is None:
			continue
		records.append({"ts": ts, "dose": float(dose), "type": bolus_type})

	df = pd.DataFrame(records)
	if not df.empty:
		df.sort_values("ts", inplace=True)
		df.reset_index(drop=True, inplace=True)
	return df


def parse_basal(root: ET.Element) -> pd.DataFrame:
	section = root.find("basal")
	records: List[Dict[str, object]] = []
	if section is None:
		return pd.DataFrame(columns=["ts", "value"])

	for event in section.findall("event"):
		ts = parse_timestamp(event.get("ts"))
		value = event.get("value")
		if ts is pd.NaT or value is None:
			continue
		records.append({"ts": ts, "value": float(value)})

	df = pd.DataFrame(records)
	if not df.empty:
		df.sort_values("ts", inplace=True)
		df.reset_index(drop=True, inplace=True)
	return df


def parse_finger_stick(root: ET.Element) -> pd.DataFrame:
	section = root.find("finger_stick")
	records: List[Dict[str, object]] = []
	if section is None:
		return pd.DataFrame(columns=["ts", "value"])

	for event in section.findall("event"):
		ts = parse_timestamp(event.get("ts"))
		value = event.get("value")
		if ts is pd.NaT or value is None:
			continue
		records.append({"ts": ts, "value": float(value)})

	df = pd.DataFrame(records)
	if not df.empty:
		df.sort_values("ts", inplace=True)
		df.reset_index(drop=True, inplace=True)
	return df


def load_patient(file_path: Path) -> PatientData:
	tree = ET.parse(file_path)
	root = tree.getroot()
	patient_id = root.get("id") or file_path.stem

	glucose = parse_glucose(root)
	meals = parse_meal(root)
	boluses = parse_bolus(root)
	basals = parse_basal(root)
	finger_sticks = parse_finger_stick(root)

	return PatientData(
		patient_id=patient_id,
		glucose=glucose,
		meals=meals,
		boluses=boluses,
		basals=basals,
		finger_sticks=finger_sticks,
	)


def describe_series(series: pd.Series) -> Dict[str, float]:
	if series.empty:
		return {"min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan, "std": np.nan}
	return {
		"min": float(series.min()),
		"max": float(series.max()),
		"mean": float(series.mean()),
		"median": float(series.median()),
		"std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
	}


def summarize_patient(data: PatientData) -> Dict[str, object]:
	glucose_stats = describe_series(data.glucose["value"])
	glucose_deltas = data.glucose["value"].diff().dropna() if not data.glucose.empty else pd.Series(dtype=float)

	meal_stats = describe_series(data.meals["carbs"]) if not data.meals.empty else describe_series(pd.Series(dtype=float))
	meals_per_day = data.meals.groupby(data.meals["ts"].dt.date).size() if not data.meals.empty else pd.Series(dtype=int)

	bolus_stats = describe_series(data.boluses["dose"]) if not data.boluses.empty else describe_series(pd.Series(dtype=float))
	bolus_per_day = data.boluses.groupby(data.boluses["ts"].dt.date).size() if not data.boluses.empty else pd.Series(dtype=int)

	basal_stats = describe_series(data.basals["value"]) if not data.basals.empty else describe_series(pd.Series(dtype=float))
	basal_changes_per_day = data.basals.groupby(data.basals["ts"].dt.date).size() if not data.basals.empty else pd.Series(dtype=int)

	finger_stats = describe_series(data.finger_sticks["value"]) if not data.finger_sticks.empty else describe_series(pd.Series(dtype=float))

	return {
		"patient_id": data.patient_id,
		"glucose_stats": glucose_stats,
		"glucose_deltas": glucose_deltas,
		"meal_stats": meal_stats,
		"meals_per_day": meals_per_day,
		"bolus_stats": bolus_stats,
		"bolus_per_day": bolus_per_day,
		"basal_stats": basal_stats,
		"basal_changes_per_day": basal_changes_per_day,
		"finger_stats": finger_stats,
		"glucose": data.glucose,
		"meals": data.meals,
		"boluses": data.boluses,
		"basals": data.basals,
		"finger_sticks": data.finger_sticks,
	}


def print_stats(title: str, stats: Dict[str, float]) -> None:
	print(f"  {title}:")
	print(f"    min    : {stats['min']:.2f}" if not np.isnan(stats["min"]) else "    min    : n/a")
	print(f"    max    : {stats['max']:.2f}" if not np.isnan(stats["max"]) else "    max    : n/a")
	print(f"    mean   : {stats['mean']:.2f}" if not np.isnan(stats["mean"]) else "    mean   : n/a")
	print(f"    median : {stats['median']:.2f}" if not np.isnan(stats["median"]) else "    median : n/a")
	print(f"    std    : {stats['std']:.2f}" if not np.isnan(stats["std"]) else "    std    : n/a")


def print_patient_summary(summary: Dict[str, object]) -> None:
	pid = summary["patient_id"]
	print(f"\nPatient {pid}")
	print_stats("Glucose", summary["glucose_stats"])

	deltas = summary["glucose_deltas"]
	if not deltas.empty:
		delta_desc = describe_series(deltas)
		print("  Delta glucose (5-min diffs):")
		print(f"    median : {delta_desc['median']:.2f}")
		print(f"    min    : {delta_desc['min']:.2f}")
		print(f"    max    : {delta_desc['max']:.2f}")
		print(f"    std    : {delta_desc['std']:.2f}")
	else:
		print("  Delta glucose (5-min diffs): n/a")

	print_stats("Meals (carbs)", summary["meal_stats"])
	meals_per_day = summary["meals_per_day"]
	if not meals_per_day.empty:
		print(f"  Meals per day (median): {meals_per_day.median():.1f} (min {meals_per_day.min()}, max {meals_per_day.max()})")
	else:
		print("  Meals per day: n/a")

	print_stats("Bolus insulin (units)", summary["bolus_stats"])
	bolus_per_day = summary["bolus_per_day"]
	if not bolus_per_day.empty:
		print(f"  Boluses per day (median): {bolus_per_day.median():.1f} (min {bolus_per_day.min()}, max {bolus_per_day.max()})")
	else:
		print("  Boluses per day: n/a")

	print_stats("Basal rate (U/hr)", summary["basal_stats"])
	basal_changes = summary["basal_changes_per_day"]
	if not basal_changes.empty:
		print(f"  Basal changes per day (median): {basal_changes.median():.1f} (min {basal_changes.min()}, max {basal_changes.max()})")
	else:
		print("  Basal changes per day: n/a")

	finger_stats = summary["finger_stats"]
	print("  Finger-stick validation:")
	if not np.isnan(finger_stats["min"]):
		print(f"    finger-stick min/max: {finger_stats['min']:.2f} / {finger_stats['max']:.2f}")
		g_stats = summary["glucose_stats"]
		if not np.isnan(g_stats["min"]):
			print(f"    CGM min/max         : {g_stats['min']:.2f} / {g_stats['max']:.2f}")
	else:
		print("    finger-stick data missing")


def aggregate_summaries(summaries: Iterable[Dict[str, object]]) -> Dict[str, object]:
	glucose_frames = [s["glucose"] for s in summaries if not s["glucose"].empty]
	meals_frames = [s["meals"] for s in summaries if not s["meals"].empty]
	bolus_frames = [s["boluses"] for s in summaries if not s["boluses"].empty]
	basal_frames = [s["basals"] for s in summaries if not s["basals"].empty]
	finger_frames = [s["finger_sticks"] for s in summaries if not s["finger_sticks"].empty]

	all_glucose = pd.concat(glucose_frames, ignore_index=True) if glucose_frames else pd.DataFrame(columns=["ts", "value"])
	all_meals = pd.concat(meals_frames, ignore_index=True) if meals_frames else pd.DataFrame(columns=["ts", "carbs", "type"])
	all_boluses = pd.concat(bolus_frames, ignore_index=True) if bolus_frames else pd.DataFrame(columns=["ts", "dose", "type"])
	all_basals = pd.concat(basal_frames, ignore_index=True) if basal_frames else pd.DataFrame(columns=["ts", "value"])
	all_fingers = pd.concat(finger_frames, ignore_index=True) if finger_frames else pd.DataFrame(columns=["ts", "value"])

	agg: Dict[str, object] = {
		"glucose_stats": describe_series(all_glucose["value"]) if not all_glucose.empty else describe_series(pd.Series(dtype=float)),
		"glucose_deltas": all_glucose["value"].diff().dropna() if not all_glucose.empty else pd.Series(dtype=float),
		"meal_stats": describe_series(all_meals["carbs"]) if not all_meals.empty else describe_series(pd.Series(dtype=float)),
		"meals_per_day": all_meals.groupby(all_meals["ts"].dt.date).size() if not all_meals.empty else pd.Series(dtype=int),
		"bolus_stats": describe_series(all_boluses["dose"]) if not all_boluses.empty else describe_series(pd.Series(dtype=float)),
		"bolus_per_day": all_boluses.groupby(all_boluses["ts"].dt.date).size() if not all_boluses.empty else pd.Series(dtype=int),
		"basal_stats": describe_series(all_basals["value"]) if not all_basals.empty else describe_series(pd.Series(dtype=float)),
		"basal_changes_per_day": all_basals.groupby(all_basals["ts"].dt.date).size() if not all_basals.empty else pd.Series(dtype=int),
		"glucose": all_glucose,
		"meals": all_meals,
		"boluses": all_boluses,
		"basals": all_basals,
		"finger_sticks": all_fingers,
	}
	return agg


def print_aggregate(agg: Dict[str, object]) -> None:
	print("\nAggregate across patients")
	print_stats("Glucose", agg["glucose_stats"])

	deltas = agg["glucose_deltas"]
	if not deltas.empty:
		delta_desc = describe_series(deltas)
		print("  Delta glucose (5-min diffs):")
		print(f"    median : {delta_desc['median']:.2f}")
		print(f"    min    : {delta_desc['min']:.2f}")
		print(f"    max    : {delta_desc['max']:.2f}")
		print(f"    std    : {delta_desc['std']:.2f}")
	else:
		print("  Delta glucose (5-min diffs): n/a")

	print_stats("Meals (carbs)", agg["meal_stats"])
	meals_per_day = agg["meals_per_day"]
	if not meals_per_day.empty:
		print(f"  Meals per day (median): {meals_per_day.median():.1f} (min {meals_per_day.min()}, max {meals_per_day.max()})")
	else:
		print("  Meals per day: n/a")

	print_stats("Bolus insulin (units)", agg["bolus_stats"])
	bolus_per_day = agg["bolus_per_day"]
	if not bolus_per_day.empty:
		print(f"  Boluses per day (median): {bolus_per_day.median():.1f} (min {bolus_per_day.min()}, max {bolus_per_day.max()})")
	else:
		print("  Boluses per day: n/a")

	print_stats("Basal rate (U/hr)", agg["basal_stats"])
	basal_changes = agg["basal_changes_per_day"]
	if not basal_changes.empty:
		print(f"  Basal changes per day (median): {basal_changes.median():.1f} (min {basal_changes.min()}, max {basal_changes.max()})")
	else:
		print("  Basal changes per day: n/a")


def plot_glucose_hist(glucose: pd.DataFrame) -> None:
	if glucose.empty:
		print("[plot] Skipping glucose histogram (no data)")
		return
	plt.figure(figsize=(8, 4))
	plt.hist(glucose["value"], bins=50, color="#4c78a8", edgecolor="black", alpha=0.8)
	plt.title("Glucose distribution")
	plt.xlabel("Glucose (mg/dL)")
	plt.ylabel("Count")
	plt.tight_layout()


def plot_glucose_timeseries(glucose: pd.DataFrame, patient_id: str, days: int) -> None:
	if glucose.empty:
		print("[plot] Skipping glucose time series (no data)")
		return
	start = glucose["ts"].min()
	window_end = start + pd.Timedelta(days=days)
	window = glucose[(glucose["ts"] >= start) & (glucose["ts"] <= window_end)]
	if window.empty:
		print("[plot] Skipping glucose time series (empty window)")
		return
	plt.figure(figsize=(10, 4))
	plt.plot(window["ts"], window["value"], color="#f58518", linewidth=1.0)
	plt.title(f"Glucose time series (first {days} day(s)) — patient {patient_id}")
	plt.xlabel("Timestamp")
	plt.ylabel("Glucose (mg/dL)")
	plt.tight_layout()


def plot_meal_or_bolus(meals: pd.DataFrame, boluses: pd.DataFrame) -> None:
	if not meals.empty:
		plt.figure(figsize=(8, 4))
		plt.hist(meals["carbs"], bins=30, color="#54a24b", edgecolor="black", alpha=0.8)
		plt.title("Meal carbohydrate distribution")
		plt.xlabel("Carbs (grams)")
		plt.ylabel("Count")
		plt.tight_layout()
		return

	if not boluses.empty:
		plt.figure(figsize=(8, 4))
		plt.hist(boluses["dose"], bins=30, color="#e45756", edgecolor="black", alpha=0.8)
		plt.title("Bolus insulin dose distribution")
		plt.xlabel("Dose (units)")
		plt.ylabel("Count")
		plt.tight_layout()
		return

	print("[plot] Skipping meal/bolus distribution (no data)")


def derive_simulator_params(agg: Dict[str, object], summaries: List[Dict[str, object]]) -> Dict[str, object]:
	"""Derive data-driven simulator parameter ranges from aggregated EDA results.

	All computations are based solely on empirical distributions parsed from the
	OhioT1DM XML files. No clinical constants are hard-coded. Finger-stick values
	are used only to validate CGM min/max and are not used to set parameters.
	"""

	# Glucose dynamics
	glucose = agg["glucose"]
	params: Dict[str, object] = {}
	if not glucose.empty:
		g_vals = glucose["value"].astype(float)
		g_min = float(np.nanmin(g_vals))
		g_max = float(np.nanmax(g_vals))
		# Observation bounds: empirical min/max observed in CGM
		obs_bounds = {"min": g_min, "max": g_max}  # data-driven bounds

		# Short-term variability: 5-minute deltas statistics
		g_deltas = agg["glucose_deltas"].astype(float)
		# Use standard deviation and 95th percentile of absolute deltas to capture typical + occasional swings
		delta_std = float(np.nanstd(g_deltas, ddof=1)) if len(g_deltas) > 1 else 0.0
		abs_deltas = np.abs(g_deltas)
		p95_abs_delta = float(np.nanpercentile(abs_deltas, 95)) if len(abs_deltas) else np.nan

		# Safe operating range: dense region via IQR (25th–75th percentiles)
		q25 = float(np.nanpercentile(g_vals, 25))
		q75 = float(np.nanpercentile(g_vals, 75))
		safe_range = {"low": q25, "high": q75}  # central mass where most readings fall

		# Rare tails for thresholds: data-driven 2.5th and 97.5th percentiles
		p2_5 = float(np.nanpercentile(g_vals, 2.5))
		p97_5 = float(np.nanpercentile(g_vals, 97.5))
		thresholds = {
			"hypoglycemia": p2_5,            # lower rare tail from empirical distribution
			"severe_hyperglycemia": p97_5,   # upper rare tail from empirical distribution
		}

		params["glucose"] = {
			"observation_bounds": obs_bounds,           # computed from empirical min/max
			"short_term_variability": {
				"delta_std": delta_std,                  # std of 5-min deltas (typical variability)
				"abs_delta_p95": p95_abs_delta,          # 95th pct of |delta| (occasional swings)
			},
			"safe_range_iqr": safe_range,              # dense region via IQR
			"tail_thresholds": thresholds,             # rare tail-based thresholds
		}

	# Meals
	meals = agg["meals"]
	if not meals.empty:
		m_vals = meals["carbs"].astype(float)
		m_med = float(np.nanmedian(m_vals))               # typical meal size
		m_q25 = float(np.nanpercentile(m_vals, 25))
		m_q75 = float(np.nanpercentile(m_vals, 75))
		m_iqr = {"low": m_q25, "high": m_q75}          # common meal range
		m_p95 = float(np.nanpercentile(m_vals, 95))       # cap excluding extreme outliers
		# Average meals/day: average the per-patient per-day counts for fairness
		per_patient_meals_day = []
		for s in summaries:
			mpd = s["meals_per_day"]
			if not mpd.empty:
				per_patient_meals_day.append(float(mpd.mean()))
		meals_per_day_avg = float(np.mean(per_patient_meals_day)) if per_patient_meals_day else np.nan

		params["meals"] = {
			"typical_median_carbs": m_med,              # median carbs per meal
			"common_range_iqr": m_iqr,                  # IQR captures usual meal sizes
			"upper_cap_p95": m_p95,                     # 95th pct excludes rare extremes
			"avg_meals_per_day": meals_per_day_avg,     # mean of per-patient daily counts
		}

	# Bolus insulin
	boluses = agg["boluses"]
	if not boluses.empty:
		b_vals = boluses["dose"].astype(float)
		b_med = float(np.nanmedian(b_vals))
		b_q25 = float(np.nanpercentile(b_vals, 25))
		b_q75 = float(np.nanpercentile(b_vals, 75))
		b_p95 = float(np.nanpercentile(b_vals, 95))
		params["bolus"] = {
			"typical_median_units": b_med,             # median bolus dose
			"common_range_iqr": {"low": b_q25, "high": b_q75},  # typical dosing scale
			"upper_bound_p95": b_p95,                  # excludes rare extreme corrections
		}

	# Basal insulin
	basals = agg["basals"]
	if not basals.empty:
		ba_vals = basals["value"].astype(float)
		ba_mean = float(np.nanmean(ba_vals))
		ba_q25 = float(np.nanpercentile(ba_vals, 25))
		ba_q75 = float(np.nanpercentile(ba_vals, 75))
		ba_std = float(np.nanstd(ba_vals, ddof=1)) if len(ba_vals) > 1 else 0.0
		cv = float(ba_std / ba_mean) if ba_mean > 0 else np.nan  # coefficient of variation
		params["basal"] = {
			"mean_rate_u_per_hr": ba_mean,            # mean basal rate across patients
			"typical_range_iqr": {"low": ba_q25, "high": ba_q75},
			"coefficient_of_variation": cv,           # low CV supports constant background
			"model_comment": "Basal modeled as constant background effect; low variance relative to bolus/meal supports this",
		}

	# Validation (finger-stick): confirm plausibility only
	fingers = agg.get("finger_sticks", pd.DataFrame(columns=["ts", "value"]))
	if not fingers.empty and not glucose.empty:
		fs_vals = fingers["value"].astype(float)
		fs_min = float(np.nanmin(fs_vals))
		fs_max = float(np.nanmax(fs_vals))
		params["validation"] = {
			"finger_stick_min_max": {"min": fs_min, "max": fs_max},
			"cgm_min_max": {"min": obs_bounds["min"], "max": obs_bounds["max"]},
			"note": "Finger-stick used only to confirm CGM min/max plausibility; not used to set parameters",
		}

	return params


def find_patient_files(data_dir: Path) -> List[Path]:
	return sorted(data_dir.glob("*.xml"))


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="EDA for OhioT1DM XML files")
	parser.add_argument("--data-dir", type=Path, default=Path("./data/raw/ohio_t1dm"), help="Directory containing patient XML files")
	parser.add_argument("--days", type=int, default=2, help="Days to show in glucose time series plot")
	parser.add_argument("--no-plots", action="store_true", help="Disable matplotlib plots")
	parser.add_argument("--output-dir", type=Path, default=Path("./data/plots_eda"), help="Directory to save plots")
	args = parser.parse_args(argv)
	
	# Create output directory if needed
	if not args.no_plots:
		args.output_dir.mkdir(parents=True, exist_ok=True)

	files = find_patient_files(args.data_dir)
	if not files:
		print(f"No XML files found in {args.data_dir}")
		return 1

	summaries: List[Dict[str, object]] = []
	for file_path in files:
		patient = load_patient(file_path)
		summary = summarize_patient(patient)
		summaries.append(summary)
		print_patient_summary(summary)

	agg = aggregate_summaries(summaries)
	print_aggregate(agg)

	if not args.no_plots:
		plot_glucose_hist(agg["glucose"])
		plt.savefig(args.output_dir / "glucose_histogram.png", dpi=100, bbox_inches="tight")
		plt.close()
		print(f"[plot] Saved glucose histogram to {args.output_dir / 'glucose_histogram.png'}")
		
		first_patient = summaries[0]
		plot_glucose_timeseries(first_patient["glucose"], first_patient["patient_id"], args.days)
		plt.savefig(args.output_dir / "glucose_timeseries.png", dpi=100, bbox_inches="tight")
		plt.close()
		print(f"[plot] Saved glucose time series to {args.output_dir / 'glucose_timeseries.png'}")
		
		plot_meal_or_bolus(agg["meals"], agg["boluses"])
		plt.savefig(args.output_dir / "meal_carb_distribution.png", dpi=100, bbox_inches="tight")
		plt.close()
		print(f"[plot] Saved meal/carb distribution to {args.output_dir / 'meal_carb_distribution.png'}")

	# Derive simulator parameters directly from EDA outputs and save
	params = derive_simulator_params(agg, summaries)
	print("\nDerived simulator parameters (data-driven):")
	# Glucose
	if "glucose" in params:
		g = params["glucose"]
		print("- Glucose observation bounds (empirical min/max):", g["observation_bounds"])  # min/max from CGM
		print("- Short-term variability: std(Δ5min)=", f"{g['short_term_variability']['delta_std']:.2f}", 
			"; |Δ5min| p95=", f"{g['short_term_variability']['abs_delta_p95']:.2f}")  # typical + occasional swings
		print("- Safe range (IQR of glucose):", g["safe_range_iqr"])  # dense distribution region
		print("- Tail thresholds (rare behavior):", g["tail_thresholds"])  # hypoglycemia / severe hyperglycemia
	# Meals
	if "meals" in params:
		m = params["meals"]
		print("- Typical meal size (median carbs):", f"{m['typical_median_carbs']:.2f}")
		print("- Common meal range (IQR):", m["common_range_iqr"])  # central mass of meals
		print("- Upper cap for meal size (p95):", f"{m['upper_cap_p95']:.2f}")  # excludes extreme outliers
		print("- Avg meals per day (per-patient mean):", f"{m['avg_meals_per_day']:.2f}")
	# Bolus
	if "bolus" in params:
		b = params["bolus"]
		print("- Typical bolus dose (median units):", f"{b['typical_median_units']:.2f}")
		print("- Common dosing scale (IQR):", b["common_range_iqr"])  # typical range
		print("- Upper bound excluding extremes (p95):", f"{b['upper_bound_p95']:.2f}")
	# Basal
	if "basal" in params:
		ba = params["basal"]
		print("- Mean basal rate (U/hr):", f"{ba['mean_rate_u_per_hr']:.2f}")
		print("- Typical basal range (IQR):", ba["typical_range_iqr"])  # observed variability
		print("- Basal CV (std/mean):", f"{ba['coefficient_of_variation']:.3f}", "=>", ba["model_comment"])  # justification
	# Validation
	if "validation" in params:
		v = params["validation"]
		print("- Validation (finger-stick min/max):", v["finger_stick_min_max"],
			"vs CGM:", v["cgm_min_max"], "[not used to set parameters]")

	# Save parameters to JSON for reproducibility from EDA outputs alone
	params_path = args.output_dir / "eda_derived_simulator_params.json"
	with open(params_path, "w") as f:
		json.dump(params, f, indent=2)
	print(f"[params] Saved derived parameters to {params_path}")


if __name__ == "__main__":
	sys.exit(main())
