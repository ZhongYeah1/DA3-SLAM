#!/usr/bin/env python3
"""
Standalone scene viewer for DA3-SLAM results.

Usage:
    python view_scene.py logs/scene_baseline/
    python view_scene.py logs/scene_baseline/ logs/scene_refine/   # multi-run comparison
"""

import sys
import os
import json
import time
import threading
import argparse
from typing import Dict, List, Optional

import numpy as np
import viser
import viser.transforms as viser_tf


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class SceneData:
    """In-memory representation of one exported scene directory."""

    def __init__(self, scene_dir: str):
        self.scene_dir = scene_dir
        self.name = os.path.basename(scene_dir.rstrip("/"))

        with open(os.path.join(scene_dir, "metadata.json")) as f:
            meta = json.load(f)
        self.strategy: str = meta.get("strategy", self.name)
        self.num_frames: int = meta["num_frames"]
        self.num_submaps: int = meta["num_submaps"]

        self.poses: np.ndarray = np.load(os.path.join(scene_dir, "poses.npy"))          # (N,4,4)
        self.intrinsics: np.ndarray = np.load(os.path.join(scene_dir, "intrinsics.npy"))# (N,3,3)

        with open(os.path.join(scene_dir, "submap_info.json")) as f:
            raw = json.load(f)
        # {frame_idx(int): submap_id(int)}
        self.submap_info: Dict[int, int] = {int(k): int(v) for k, v in raw.items()}

        # Unique submap ids in temporal order
        seen = []
        for i in range(self.num_frames):
            sid = self.submap_info[i]
            if sid not in seen:
                seen.append(sid)
        self.submap_ids: List[int] = seen

        # Load all frames
        frames_dir = os.path.join(scene_dir, "frames")
        self.points: List[np.ndarray] = []   # each (H,W,3) float32
        self.colors: List[np.ndarray] = []   # each (H,W,3) uint8
        self.conf:   List[np.ndarray] = []   # each (H,W) float32

        for i in range(self.num_frames):
            npz = np.load(os.path.join(frames_dir, f"{i:04d}.npz"))
            self.points.append(npz["points"])
            self.colors.append(npz["colors"])
            self.conf.append(npz["conf"])

        # Global conf range for slider normalisation
        all_conf = np.concatenate([c.ravel() for c in self.conf])
        self.conf_min = float(all_conf.min())
        self.conf_max = float(all_conf.max())

        print(f"[viewer] Loaded '{self.name}': {self.num_frames} frames, "
              f"{self.num_submaps} submaps")


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

class SceneViewer:
    def __init__(self, scene_dirs: List[str], port: int = 8080):
        # Validate all scene directories exist before attempting to load anything
        for p in scene_dirs:
            if not os.path.isdir(p):
                raise FileNotFoundError(f"Scene directory not found: {p}")

        # Load all scenes
        self.scenes: List[SceneData] = [SceneData(d) for d in scene_dirs]
        self.active_run_idx: int = 0

        # Viser server
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        # Scene handles: indexed by (run_idx, frame_idx)
        self._pcd_handles: Dict[tuple, object] = {}
        self._frustum_handles: Dict[tuple, List[object]] = {}

        # Per-submap colours (same seed as viewer.py)
        np.random.seed(100)
        self._submap_colors = np.random.randint(0, 256, size=(250, 3), dtype=np.uint8)

        # Playback state
        self._play_event  = threading.Event()   # set = playing
        self._stop_event  = threading.Event()   # set = stop thread
        self._reset_event = threading.Event()   # set = jump to frame 0
        self._play_thread: Optional[threading.Thread] = None

        # Build GUI
        self._build_gui()

        # Render initial scene
        self._render_active_scene()

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------

    def _build_gui(self):
        scene = self.scenes[self.active_run_idx]

        # --- Run selector (only shown when >1 run) ---
        run_names = [s.name for s in self.scenes]
        if len(self.scenes) > 1:
            self._gui_run = self.server.gui.add_dropdown(
                "Run", options=run_names, initial_value=run_names[0]
            )
            self._gui_run.on_update(self._on_run_change)
        else:
            self._gui_run = None

        # --- Confidence slider ---
        # Display as percentile [0, 100]; maps to actual conf value at render time
        self._gui_conf = self.server.gui.add_slider(
            "Conf threshold %", min=0, max=100, step=1, initial_value=25
        )
        self._gui_conf.on_update(self._on_conf_change)

        # --- Show cameras ---
        self._gui_show_cams = self.server.gui.add_checkbox("Show cameras", initial_value=True)
        self._gui_show_cams.on_update(self._on_show_cams_change)

        # --- Playback controls ---
        self._gui_play  = self.server.gui.add_button("Play")
        self._gui_pause = self.server.gui.add_button("Pause")
        self._gui_reset = self.server.gui.add_button("Reset")
        self._gui_speed = self.server.gui.add_slider(
            "Playback FPS", min=1, max=60, step=1, initial_value=20
        )
        self._gui_play.on_click(self._on_play)
        self._gui_pause.on_click(self._on_pause)
        self._gui_reset.on_click(self._on_reset)

        # --- Screenshot ---
        self._gui_screenshot = self.server.gui.add_button("Screenshot")
        self._gui_screenshot.on_click(self._on_screenshot)

        # --- Per-submap checkboxes (built fresh in _render_active_scene) ---
        self._submap_checkboxes: Dict[int, object] = {}

    def _rebuild_submap_checkboxes(self, scene: SceneData):
        """Tear down old checkboxes and create new ones for active scene."""
        # Viser doesn't expose remove_gui, so we just overwrite the dict;
        # old handles become unreferenced. This is acceptable for a viewer.
        self._submap_checkboxes = {}
        for sid in scene.submap_ids:
            cb = self.server.gui.add_checkbox(f"Submap {sid}", initial_value=True)
            cb.on_update(lambda _, s=sid: self._on_submap_toggle(s))
            self._submap_checkboxes[sid] = cb

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _conf_threshold_value(self, scene: SceneData, percentile: float) -> float:
        """Convert slider percentile to absolute conf value."""
        all_conf = np.concatenate([c.ravel() for c in scene.conf])
        return float(np.percentile(all_conf, percentile))

    def _render_active_scene(self):
        """Clear all scene objects and render the currently active run."""
        # Remove all existing point clouds and frustums
        for handle in self._pcd_handles.values():
            handle.remove()
        for handles in self._frustum_handles.values():
            for h in handles:
                h.remove()
        self._pcd_handles.clear()
        self._frustum_handles.clear()

        scene = self.scenes[self.active_run_idx]
        self._rebuild_submap_checkboxes(scene)
        conf_thresh = self._conf_threshold_value(scene, self._gui_conf.value)

        for frame_idx in range(scene.num_frames):
            submap_id = scene.submap_info[frame_idx]
            self._render_frame(scene, frame_idx, submap_id, conf_thresh)

    def _render_frame(self, scene: SceneData, frame_idx: int, submap_id: int, conf_thresh: float):
        run_idx = self.active_run_idx
        color = self._submap_colors[submap_id % len(self._submap_colors)]

        # --- Point cloud ---
        points_hw3 = scene.points[frame_idx]    # (H,W,3)
        colors_hw3 = scene.colors[frame_idx]    # (H,W,3)
        conf_hw    = scene.conf[frame_idx]       # (H,W)
        mask = conf_hw >= conf_thresh
        pts_flat = points_hw3[mask]              # (M,3)
        col_flat = colors_hw3[mask]              # (M,3) uint8

        if pts_flat.shape[0] > 0:
            pcd_handle = self.server.scene.add_point_cloud(
                name=f"run{run_idx}/submap{submap_id}/pcd{frame_idx}",
                points=pts_flat.astype(np.float32),
                colors=col_flat,
                point_size=0.001,
                point_shape="circle",
            )
            self._pcd_handles[(run_idx, frame_idx)] = pcd_handle

        # --- Camera frustum ---
        c2w_44 = scene.poses[frame_idx]           # (4,4)
        T_world_camera = viser_tf.SE3.from_matrix(c2w_44[:3, :])  # SE3 from 3x4

        K = scene.intrinsics[frame_idx]            # (3,3)
        # Reconstruct a representative image thumbnail from colors (downsampled)
        thumb = colors_hw3[::4, ::4]               # (H/4, W/4, 3)
        H_t, W_t = thumb.shape[:2]
        fy = K[1, 1]
        fov = 2 * np.arctan2(H_t * 2, fy)         # approx fov from full-res fy

        frustum_name = f"run{run_idx}/submap{submap_id}/frustum{frame_idx}"
        frame_name   = f"run{run_idx}/submap{submap_id}/frame{frame_idx}"

        frame_handle = self.server.scene.add_frame(
            frame_name,
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
            axes_length=0.05,
            axes_radius=0.002,
            origin_radius=0.002,
            visible=self._gui_show_cams.value,
        )
        frustum_handle = self.server.scene.add_camera_frustum(
            frustum_name,
            fov=fov,
            aspect=W_t / H_t,
            scale=0.05,
            image=thumb,
            line_width=2.0,
            color=color,
            visible=self._gui_show_cams.value,
        )
        self._frustum_handles[(run_idx, frame_idx)] = [frame_handle, frustum_handle]

    def _refresh_point_clouds(self):
        """Re-render all point clouds with updated confidence threshold."""
        scene = self.scenes[self.active_run_idx]
        conf_thresh = self._conf_threshold_value(scene, self._gui_conf.value)
        run_idx = self.active_run_idx

        for frame_idx in range(scene.num_frames):
            submap_id = scene.submap_info[frame_idx]
            key = (run_idx, frame_idx)

            # Remove old cloud
            if key in self._pcd_handles:
                self._pcd_handles[key].remove()
                del self._pcd_handles[key]

            # Check if submap is toggled on
            cb = self._submap_checkboxes.get(submap_id)
            if cb is not None and not cb.value:
                continue

            points_hw3 = scene.points[frame_idx]
            colors_hw3 = scene.colors[frame_idx]
            conf_hw    = scene.conf[frame_idx]
            mask = conf_hw >= conf_thresh
            pts_flat = points_hw3[mask]
            col_flat = colors_hw3[mask]

            if pts_flat.shape[0] > 0:
                self._pcd_handles[key] = self.server.scene.add_point_cloud(
                    name=f"run{run_idx}/submap{submap_id}/pcd{frame_idx}",
                    points=pts_flat.astype(np.float32),
                    colors=col_flat,
                    point_size=0.001,
                    point_shape="circle",
                )

    # ------------------------------------------------------------------
    # GUI callbacks
    # ------------------------------------------------------------------

    def _on_run_change(self, _):
        if self._gui_run is None:
            return
        new_name = self._gui_run.value
        for i, s in enumerate(self.scenes):
            if s.name == new_name:
                self.active_run_idx = i
                break
        self._render_active_scene()

    def _on_conf_change(self, _):
        self._refresh_point_clouds()

    def _on_show_cams_change(self, _):
        visible = self._gui_show_cams.value
        run_idx = self.active_run_idx
        for frame_idx in range(self.scenes[run_idx].num_frames):
            for handle in self._frustum_handles.get((run_idx, frame_idx), []):
                handle.visible = visible

    def _on_submap_toggle(self, submap_id: int):
        scene = self.scenes[self.active_run_idx]
        run_idx = self.active_run_idx
        cb = self._submap_checkboxes[submap_id]
        visible = cb.value

        for frame_idx, sid in scene.submap_info.items():
            if sid != submap_id:
                continue
            # Toggle point cloud
            pcd = self._pcd_handles.get((run_idx, frame_idx))
            if pcd is not None:
                pcd.visible = visible
            # Toggle frustums
            for handle in self._frustum_handles.get((run_idx, frame_idx), []):
                handle.visible = visible

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _on_play(self, _):
        if self._play_thread is not None and self._play_thread.is_alive():
            self._play_event.set()  # resume if paused
            return
        self._stop_event.clear()
        self._reset_event.clear()
        self._play_event.set()
        self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._play_thread.start()

    def _on_pause(self, _):
        self._play_event.clear()

    def _on_reset(self, _):
        self._stop_event.set()
        self._play_event.clear()

    def _playback_loop(self):
        # Re-capture scene at the start of each frame to pick up run switches.
        # Without this, a run change mid-playback would still iterate the old
        # scene's frame count and camera poses.
        scene = self.scenes[self.active_run_idx]
        for frame_idx in range(scene.num_frames):
            scene = self.scenes[self.active_run_idx]  # re-capture each iteration
            # Stop check
            if self._stop_event.is_set():
                return
            # Pause: wait until play is set again or stop is triggered
            while not self._play_event.is_set():
                if self._stop_event.is_set():
                    return
                time.sleep(0.05)

            c2w = scene.poses[frame_idx]
            T = viser_tf.SE3.from_matrix(c2w[:3, :])
            wxyz = T.rotation().wxyz
            pos  = T.translation()

            clients = self.server.get_clients()
            for client in clients.values():
                client.camera.wxyz     = wxyz
                client.camera.position = pos

            fps = max(1, self._gui_speed.value)
            time.sleep(1.0 / fps)

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    def _on_screenshot(self, _):
        clients = self.server.get_clients()
        if not clients:
            print("[viewer] No clients connected.")
            return
        # Request screenshot from first connected client
        client = next(iter(clients.values()))
        try:
            client.request_screenshot().result(timeout=10.0)  # blocks until delivered
        except Exception as e:
            print(f"[viewer] Screenshot timed out or failed: {e}")

    # ------------------------------------------------------------------
    # Block until server exits
    # ------------------------------------------------------------------

    def run(self):
        print(f"[viewer] Serving on http://0.0.0.0:{self.server.get_port()}")
        print("[viewer] Press Ctrl-C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DA3-SLAM standalone scene viewer")
    parser.add_argument("scene_dirs", nargs="+", metavar="SCENE_DIR",
                        help="One or more exported scene directories")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    viewer = SceneViewer(scene_dirs=args.scene_dirs, port=args.port)
    viewer.run()


if __name__ == "__main__":
    main()
