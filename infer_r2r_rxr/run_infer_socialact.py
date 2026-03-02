#!/usr/bin/env python3
"""
SocialACT Dataset Evaluation using OmniNav

Evaluates the SocialACT dataset with per-sub-mission instruction switching.
For each mission, sub-missions are evaluated sequentially with agent reset
between sub-missions. Results (CSV + MP4) are accumulated and saved per mission.

Two evaluation rounds are run automatically:
  1) Low-level instructions  -> ./SocaiACT_Test/Low-level_instruction/
  2) High-level instructions -> ./SocaiACT_Test/High-level_instruction/

Usage:
    python run_infer_socialact.py --model-path ../OmniNav --dataset-dir ../SocialACT
"""

# HPC-X/UCC library conflict prevention (NGC container issue)
import os
import sys

_LD_PRELOAD_LIBS = "/opt/hpcx/ucx/lib/libucs.so.0:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucm.so.0"
_REEXEC_VAR = "_OMNINAV_REEXEC"

if os.environ.get(_REEXEC_VAR) != "1" and os.path.exists("/opt/hpcx/ucx/lib/libucs.so.0"):
    os.environ["LD_PRELOAD"] = _LD_PRELOAD_LIBS
    os.environ[_REEXEC_VAR] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import argparse
import torch
import csv
import time
import cv2
import math
import tempfile
import glob
from datetime import datetime

import openpyxl
import matplotlib.cm as mpl_cm

from agent.waypoint_agent import Waypoint_Agent


# ---------------------------------------------------------------------------
#  Helper functions (identical logic to run_infer_online_panorama.py)
# ---------------------------------------------------------------------------

def draw_waypoint_arrows_fpv(
    img: np.ndarray,
    waypoints: list,
    arrow_thickness: int = 2,
    tipLength: float = 0.45,
    stop_color: tuple = (0, 0, 255),
    stop_radius: int = 8,
    arrow_scale: float = 0.15,
    vis_scale: float = 120.0,
    arrow_gap: int = 1.2,
) -> np.ndarray:
    """Draw each waypoint as an arrow stacked vertically (same as run_infer_online_panorama.py)."""
    out = img.copy()
    h, w = out.shape[:2]
    base_x, base_y = w // 2, int(h * 0.95)

    slot_height = max(1, int(vis_scale * arrow_scale) + arrow_gap)

    try:
        cmap = mpl_cm.get_cmap('turbo')
    except Exception:
        cmap = mpl_cm.get_cmap('viridis')

    n_wp = len(waypoints)
    for i, wp in enumerate(waypoints):
        dx_net = wp.get('dx', 0.0)
        dy_net = wp.get('dy', 0.0)
        arrive = wp.get('arrive', 0)

        start_y = int(base_y - i * slot_height)
        start_pixel = (base_x, start_y)

        if arrive > 0:
            arrive_color = (0, 200, 0)
            cv2.circle(out, start_pixel, stop_radius, arrive_color, -1)
            cv2.circle(out, start_pixel, stop_radius, (255, 255, 255), 1)
            cv2.putText(out, "OK", (start_pixel[0] - 12, start_pixel[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            continue

        # Coordinate transform: Network -> Robot Body ISO 8855
        target_x = dy_net    # Net Forward -> Robot X (Forward)
        target_y = -dx_net   # Net Right   -> Robot -Y (Left)

        theta = math.atan2(target_y, target_x) if (target_x != 0 or target_y != 0) else 0.0
        body_dx = arrow_scale * math.cos(theta)
        body_dy = arrow_scale * math.sin(theta)

        head_x = int(start_pixel[0] - body_dy * vis_scale)
        head_y = int(start_pixel[1] - body_dx * vis_scale)
        head_pixel = (np.clip(head_x, 0, w - 1), np.clip(head_y, 0, h - 1))

        t = (i + 0.5) / n_wp if n_wp > 0 else 0.5
        rgba = cmap(t)[:3]
        color = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))

        if np.linalg.norm(np.array(head_pixel) - np.array(start_pixel)) > 2:
            cv2.arrowedLine(out, start_pixel, head_pixel, color, arrow_thickness,
                            tipLength=tipLength, line_type=cv2.LINE_AA)

    return out


def add_instruction_bar(img_rgb: np.ndarray, display_text: str, bar_height: int = 80) -> np.ndarray:
    """Append instruction/sub-instruction text bar below image. Returns RGB image.

    Unlike run_infer_online_panorama.py, this version uses display_text as-is
    (no 'Instruction: ' prefix) so the caller can pass 'Sub-InstructionN: ...' directly.
    """
    h, w = img_rgb.shape[:2]
    new_img = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
    new_img.fill(255)
    new_img[:h, :w] = img_rgb

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    full_text = display_text or ""
    (_, line_h), _ = cv2.getTextSize("Ay", font, font_scale, thickness)
    margin_x, margin_y = 10, 10
    max_line_w = w - 2 * margin_x

    # Word wrap
    words = full_text.split()
    lines = []
    line = ""
    for word in words:
        test = (line + " " + word) if line else word
        (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if tw <= max_line_w:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)

    y = h + margin_y + line_h
    for i, ln in enumerate(lines[:4]):
        dy = y + i * (line_h + 4)
        if dy > h + bar_height - 5:
            break
        cv2.putText(new_img, ln, (margin_x, dy),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return new_img


# ---------------------------------------------------------------------------
#  SocialACT-specific utilities
# ---------------------------------------------------------------------------

def parse_instruction_xlsx(xlsx_path):
    """Parse a SocialACT instruction Excel file.

    Excel column layout (0-indexed):
        0: Mission, 1: Goal, 2: Sub-mission_number,
        3: StartTime1, 4: Sub-Instruction1,
        5: StartTime2, 6: Sub-Instruction2,
        7: StartTime3, 8: Sub-Instruction3,
        9: StartTime4, 10: Sub-Instruction4

    Returns:
        dict  {mission_name: [(start_time_sec, instruction_text), ...]}
    """
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active

    missions = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        mission_name = row[0]
        if mission_name is None:
            continue

        num_subs = int(row[2])
        sub_missions = []
        for i in range(num_subs):
            start_time_col = 3 + i * 2
            instruction_col = 4 + i * 2

            if start_time_col >= len(row) or instruction_col >= len(row):
                break
            start_time = row[start_time_col]
            instruction = row[instruction_col]
            if start_time is not None and instruction is not None:
                sub_missions.append((float(start_time), str(instruction)))

        missions[mission_name] = sub_missions

    wb.close()
    return missions


def get_last_frame_idx(mission_dir):
    """Return the highest frame index in a mission directory."""
    frame_dirs = sorted(glob.glob(os.path.join(mission_dir, "frame_*")))
    if not frame_dirs:
        return 0
    last_name = os.path.basename(frame_dirs[-1])  # e.g. "frame_0242"
    return int(last_name.split("_")[1])


def load_frame_images(mission_dir, frame_idx):
    """Load front/left/right images for frame_XXXX as RGB numpy arrays.

    Returns:
        (front_rgb, left_rgb, right_rgb) or (None, None, None) on failure.
    """
    frame_name = f"frame_{frame_idx:04d}"
    frame_dir = os.path.join(mission_dir, frame_name)

    front_path = os.path.join(frame_dir, f"{frame_name}_front.jpg")
    left_path = os.path.join(frame_dir, f"{frame_name}_left.jpg")
    right_path = os.path.join(frame_dir, f"{frame_name}_right.jpg")

    if not all(os.path.exists(p) for p in [front_path, left_path, right_path]):
        return None, None, None

    front_bgr = cv2.imread(front_path)
    left_bgr = cv2.imread(left_path)
    right_bgr = cv2.imread(right_path)

    if front_bgr is None or left_bgr is None or right_bgr is None:
        return None, None, None

    front = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB)
    left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    return front, left, right


# ---------------------------------------------------------------------------
#  Main evaluator
# ---------------------------------------------------------------------------

class SocialACTEvaluator:
    """Evaluate the SocialACT dataset using OmniNav.

    The model is loaded once; between sub-missions only agent.reset() is
    called (clears history / KV-cache) without reloading the checkpoint.
    """

    def __init__(self, model_path, dataset_dir):
        self.dataset_dir = dataset_dir
        self.model_path = model_path

        # Create a throwaway directory for agent internals (render_img etc.)
        self._temp_agent_path = tempfile.mkdtemp(prefix="omninav_socialact_")

        print("=" * 60)
        print("[SocialACT] Loading OmniNav model (one-time) ...")
        print("=" * 60)
        self.agent = Waypoint_Agent(model_path, self._temp_agent_path, require_map=False)
        self.agent.reset()
        print("[SocialACT] Model loaded successfully.\n")

    # ------------------------------------------------------------------
    #  Frame range computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sub_mission_ranges(sub_missions, last_frame_idx):
        """Convert (start_time, instruction) list to (start_frame, end_frame, instruction).

        Rules (from dataset design):
          - start_frame = int(start_time * 2)    (2 Hz sampling)
          - end_frame   = int((next_start_time - 3) * 2)  for non-last sub-missions
          - end_frame   = last_frame_idx                   for the last sub-mission
        """
        ranges = []
        n = len(sub_missions)
        for i, (start_time, instruction) in enumerate(sub_missions):
            start_frame = int(start_time * 2)
            if i + 1 < n:
                next_start_time = sub_missions[i + 1][0]
                end_frame = int((next_start_time - 3) * 2)
            else:
                end_frame = last_frame_idx

            # Clamp to valid range
            end_frame = min(end_frame, last_frame_idx)
            if end_frame < start_frame:
                print(f"  [WARN] Sub-mission {i+1}: end_frame ({end_frame}) < start_frame ({start_frame}), skipping")
                continue

            ranges.append((start_frame, end_frame, instruction))
        return ranges

    # ------------------------------------------------------------------
    #  Waypoint list builder (matches run_infer_online_panorama.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_waypoint_list(action):
        """Extract list of waypoint dicts from agent action output."""
        if 'arrive_pred' not in action or 'action' not in action or 'recover_angle' not in action:
            return None

        arrive = int(action['arrive_pred'])
        waypoints = action['action']
        recover_angles = action['recover_angle']

        if isinstance(waypoints, np.ndarray) and waypoints.ndim > 1:
            waypoints = waypoints.reshape(-1, 2)
        if isinstance(recover_angles, np.ndarray) and recover_angles.ndim > 1:
            recover_angles = recover_angles.flatten()

        result = []
        for i in range(min(5, len(waypoints))):
            dx = float(waypoints[i][0])
            dy = float(waypoints[i][1])
            dtheta = float(np.degrees(recover_angles[i])) if i < len(recover_angles) else 0.0
            result.append({'dx': dx, 'dy': dy, 'dtheta': dtheta, 'arrive': arrive})
        return result

    # ------------------------------------------------------------------
    #  Per-mission evaluation
    # ------------------------------------------------------------------

    def evaluate_mission(self, mission_name, sub_missions, output_dir):
        """Evaluate a single mission across all its sub-missions.

        Accumulates CSV records and visualisation frames, then writes
        exactly one CSV and one MP4 per mission.
        """
        mission_dir = os.path.join(self.dataset_dir, mission_name)
        if not os.path.isdir(mission_dir):
            print(f"[WARN] Mission directory not found: {mission_dir}, skipping")
            return

        last_frame_idx = get_last_frame_idx(mission_dir)
        sub_ranges = self.compute_sub_mission_ranges(sub_missions, last_frame_idx)

        # ---- Mission start log ----
        total_eval_frames = sum(e - s + 1 for s, e, _ in sub_ranges)
        print(f"\n{'*' * 60}")
        print(f"  MISSION START: {mission_name}")
        print(f"  Total sub-missions: {len(sub_ranges)}")
        print(f"  Total frames (last): frame_{last_frame_idx:04d}")
        print(f"  Total evaluation frames: {total_eval_frames}")
        print(f"{'*' * 60}")

        PREDICT_SCALE = 0.3
        info = {'top_down_map_vlnce': None, 'gt_map': None, 'pred_map': None}
        default_pose = {'position': [0.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0]}

        # Accumulators (across all sub-missions in this mission)
        csv_records = []
        vis_frame_list = []        # (left, front_vis, right) tuples
        vis_instructions = []      # display text per frame
        total_infer_time = 0.0
        global_frame_count = 0

        for sub_idx, (start_frame, end_frame, instruction) in enumerate(sub_ranges):
            sub_label = f"Sub-Instruction{sub_idx + 1}: {instruction}"

            print(f"\n  {'─' * 56}")
            print(f"  [{mission_name}] Sub-mission {sub_idx + 1}/{len(sub_ranges)} START")
            print(f"  [{mission_name}] {sub_label}")
            print(f"  [{mission_name}] Frame range: frame_{start_frame:04d} ~ frame_{end_frame:04d}"
                  f"  ({end_frame - start_frame + 1} frames)")
            print(f"  [{mission_name}] Agent reset (history + KV cache cleared)")
            print(f"  {'─' * 56}")

            # Reset agent state (history / KV-cache); model weights are NOT reloaded
            self.agent.reset()
            self.agent.episode_id = f"{mission_name}_sub{sub_idx + 1}"

            for frame_idx in range(start_frame, end_frame + 1):
                front, left, right = load_frame_images(mission_dir, frame_idx)
                if front is None:
                    print(f"  [WARN] Missing frame_{frame_idx:04d}, skipping")
                    continue

                obs = {
                    'front': front,
                    'left': left,
                    'right': right,
                    'rgb': front,
                    'instruction': {'text': instruction},
                    'pose': default_pose,
                }

                t0 = time.time()
                with torch.no_grad():
                    action = self.agent.act(obs, info, mission_name)
                infer_time = time.time() - t0
                total_infer_time += infer_time
                global_frame_count += 1

                # ---- CSV recording (same format as run_infer_online_panorama.py) ----
                if 'arrive_pred' in action and 'action' in action and 'recover_angle' in action:
                    arrive = int(action['arrive_pred'])
                    waypoints = action['action']
                    recover_angles = action['recover_angle']
                    if isinstance(waypoints, np.ndarray) and waypoints.ndim > 1:
                        waypoints = waypoints.reshape(-1, 2)
                    if isinstance(recover_angles, np.ndarray) and recover_angles.ndim > 1:
                        recover_angles = recover_angles.flatten()

                    for sf_idx in range(min(5, len(waypoints))):
                        dx = waypoints[sf_idx][0] / PREDICT_SCALE
                        dy = waypoints[sf_idx][1] / PREDICT_SCALE
                        dtheta = (np.degrees(recover_angles[sf_idx])
                                  if sf_idx < len(recover_angles) else 0.0)
                        csv_records.append({
                            'frame_idx': frame_idx,
                            'subframe_idx': sf_idx,
                            'dx': float(dx),
                            'dy': float(dy),
                            'dtheta': float(dtheta),
                            'arrive': arrive,
                            'infer_time_s': float(infer_time) if sf_idx == 0 else 0.0,
                        })

                    wp0 = waypoints[0]
                    dtheta0 = np.degrees(recover_angles[0]) if len(recover_angles) > 0 else 0.0
                    print(f"  [frame_{frame_idx:04d}] infer={infer_time:.3f}s  arrive={arrive}"
                          f"  wp0=({wp0[0]:.4f},{wp0[1]:.4f})  dtheta={dtheta0:.2f}")

                # ---- Visualisation frame ----
                waypoint_list = self._build_waypoint_list(action)
                if waypoint_list:
                    front_vis = draw_waypoint_arrows_fpv(front, waypoint_list)
                else:
                    front_vis = front.copy()

                vis_frame_list.append((left.copy(), front_vis, right.copy()))
                vis_instructions.append(sub_label)

        # ---- Save accumulated results for this mission ----
        if csv_records:
            self._save_csv(csv_records, output_dir, mission_name)
        else:
            print(f"[{mission_name}] No CSV records to save.")

        if vis_frame_list:
            self._save_video(vis_frame_list, vis_instructions, output_dir, mission_name)
        else:
            print(f"[{mission_name}] No visualisation frames to save.")

        # Timing summary
        if global_frame_count > 0:
            avg = total_infer_time / global_frame_count
            print(f"\n[{mission_name}] Inference stats: total={total_infer_time:.1f}s, "
                  f"avg={avg:.3f}s/frame, FPS={1.0 / avg:.2f}")

    # ------------------------------------------------------------------
    #  Output helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_csv(records, output_dir, mission_name):
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{mission_name}.csv")
        fieldnames = ['frame_idx', 'subframe_idx', 'dx', 'dy', 'dtheta', 'arrive', 'infer_time_s']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        n_frames = len([r for r in records if r['subframe_idx'] == 0])
        print(f"[{mission_name}] CSV saved: {csv_path}  ({len(records)} records, {n_frames} frames)")

    @staticmethod
    def _save_video(vis_frame_list, vis_instructions, output_dir, mission_name):
        """Save MP4: 3-column layout [LEFT | FRONT(arrows) | RIGHT] + instruction bar."""
        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, f"{mission_name}.mp4")

        left_f, front_f, right_f = vis_frame_list[0]
        target_h, target_w = front_f.shape[:2]

        bar_height = 80
        out_w = target_w * 3
        out_h = target_h + bar_height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 2.0  # match the 2 Hz sampling rate of SocialACT
        writer = cv2.VideoWriter(video_path, fourcc, fps, (out_w, out_h))

        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(video_path, fourcc, fps, (out_w, out_h))

        if not writer.isOpened():
            print(f"[{mission_name}] ERROR: Could not create video: {video_path}")
            return

        for i, (lf, ff, rf) in enumerate(vis_frame_list):
            if lf.shape[:2] != (target_h, target_w):
                lf = cv2.resize(lf, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if ff.shape[:2] != (target_h, target_w):
                ff = cv2.resize(ff, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if rf.shape[:2] != (target_h, target_w):
                rf = cv2.resize(rf, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            combined = np.concatenate([lf, ff, rf], axis=1)
            disp_text = vis_instructions[i] if i < len(vis_instructions) else ""
            frame_with_bar = add_instruction_bar(combined, disp_text, bar_height)
            frame_bgr = cv2.cvtColor(frame_with_bar, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        print(f"[{mission_name}] Video saved: {video_path}  "
              f"({len(vis_frame_list)} frames, {out_w}x{out_h}, {fps} fps)")

    # ------------------------------------------------------------------
    #  Top-level driver
    # ------------------------------------------------------------------

    def evaluate_all(self, instruction_xlsx, output_dir):
        """Evaluate all SocialACT missions with instructions from *instruction_xlsx*."""
        instructions = parse_instruction_xlsx(instruction_xlsx)

        # Iterate over mission folders present on disk (sorted)
        mission_dirs = sorted([
            d for d in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, d)) and d.startswith("SocialACT_")
        ])

        print(f"\n[SocialACT] Found {len(mission_dirs)} mission folders")
        print(f"[SocialACT] Instructions loaded for: {sorted(instructions.keys())}")
        print(f"[SocialACT] Output directory: {os.path.abspath(output_dir)}\n")

        for mission_name in mission_dirs:
            if mission_name not in instructions:
                print(f"[WARN] No instructions for {mission_name} in {instruction_xlsx}, skipping")
                continue

            # Skip missions that already have both CSV and MP4 outputs
            csv_path = os.path.join(output_dir, f"{mission_name}.csv")
            mp4_path = os.path.join(output_dir, f"{mission_name}.mp4")
            if os.path.exists(csv_path) and os.path.exists(mp4_path):
                print(f"[SKIP] {mission_name}: already completed "
                      f"({os.path.basename(csv_path)}, {os.path.basename(mp4_path)})")
                continue

            print(f"\n{'#' * 60}")
            print(f"# Mission: {mission_name}  "
                  f"({len(instructions[mission_name])} sub-missions)")
            print(f"{'#' * 60}")

            self.evaluate_mission(mission_name, instructions[mission_name], output_dir)

        print(f"\n[SocialACT] All missions evaluated.  Results in: {os.path.abspath(output_dir)}")


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SocialACT dataset with OmniNav (Low-level & High-level instructions)')

    parser.add_argument("--model-path", type=str, default="../OmniNav",
                        help="Path to the OmniNav model checkpoint (default: ../OmniNav)")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="SocialACT dataset directory (default: ../SocialACT relative to this script)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output directory (default: ../SocaiACT_Test relative to this script)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = (args.dataset_dir
                   if args.dataset_dir is not None
                   else os.path.normpath(os.path.join(script_dir, "..", "SocialACT")))
    output_base = (args.output_dir
                   if args.output_dir is not None
                   else os.path.normpath(os.path.join(script_dir, "..", "SocaiACT_Test")))

    low_xlsx = os.path.join(dataset_dir, "Low-level_instruction.xlsx")
    high_xlsx = os.path.join(dataset_dir, "High-level_instruction.xlsx")

    # Validate paths
    for p, label in [(dataset_dir, "Dataset dir"), (low_xlsx, "Low-level xlsx"), (high_xlsx, "High-level xlsx")]:
        if not os.path.exists(p):
            print(f"[ERROR] {label} not found: {p}")
            sys.exit(1)

    # ---------- Build evaluator (loads model once) ----------
    evaluator = SocialACTEvaluator(args.model_path, dataset_dir)

    # ---------- Round 1: Low-level instructions ----------
    print("\n" + "=" * 80)
    print("  ROUND 1 / 2 :  LOW-LEVEL INSTRUCTIONS")
    print("=" * 80)
    low_output = os.path.join(output_base, "Low-level_instruction")
    evaluator.evaluate_all(low_xlsx, low_output)

    # ---------- Round 2: High-level instructions ----------
    print("\n" + "=" * 80)
    print("  ROUND 2 / 2 :  HIGH-LEVEL INSTRUCTIONS")
    print("=" * 80)
    high_output = os.path.join(output_base, "High-level_instruction")
    evaluator.evaluate_all(high_xlsx, high_output)

    # ---------- Summary ----------
    print("\n" + "=" * 80)
    print("  ALL EVALUATIONS COMPLETE")
    print(f"  Low-level  results -> {os.path.abspath(low_output)}")
    print(f"  High-level results -> {os.path.abspath(high_output)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
