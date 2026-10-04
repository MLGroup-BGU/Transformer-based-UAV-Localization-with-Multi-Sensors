"""
UAV & Multi-Sensor Experiment Visualizer
----------------------------------------
An interactive visualization tool to illustrate experiments described in CSV files:
- 2D layout of Sensors and UAV with Z-axis color coding.
- Timestep navigation via Slider, Prev/Next buttons, and Auto-Play animation.
- Rich hover tooltips displaying Sensor details, Drone parameters, and Detection summary.
- Fast experiment switching via Exp ID input.
- Optional HTML export (--html) for browser visualization.

Author: Antigravity Agent
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.cm as cm
import matplotlib.colors as mcolors


class ExperimentVisualizer:
    def __init__(self, data_dir, exp_id=1):
        self.data_dir = data_dir
        self.exp_id = exp_id

        # File paths
        self.sensor_info_path = os.path.join(data_dir, "sensor_information.csv")
        self.sensor_dist_path = os.path.join(data_dir, "sensor_distances.csv")
        self.drone_info_path = os.path.join(data_dir, "drone_information.csv")
        self.drone_traj_path = os.path.join(data_dir, "drone_trajectories.csv")
        self.sensor_det_path = os.path.join(data_dir, "sensor_detection_summary.csv")

        # Global Dataframes
        self.df_sensor_info = None
        self.df_sensor_dist = None
        self.df_drone_info = None
        self.df_drone_traj = None
        self.df_sensor_det = None

        # Current Experiment Cache
        self.current_exp_sensors = None
        self.current_exp_drone_info = None
        self.current_exp_traj = None
        self.current_exp_det = None

        # State
        self.current_step = 0
        self.is_playing = False
        self.timer = None

        self.preload_data()
        self.load_experiment(exp_id)

    def preload_data(self):
        """Pre-load all CSV files into memory for instant filtering."""
        print("==================================================")
        print(f"Loading experiment datasets from:\n{self.data_dir}")
        print("==================================================")
        t0 = time.time()

        if os.path.exists(self.sensor_info_path):
            print("  * Loading sensor_information.csv...")
            self.df_sensor_info = pd.read_csv(self.sensor_info_path)

        if os.path.exists(self.sensor_dist_path):
            print("  * Loading sensor_distances.csv...")
            self.df_sensor_dist = pd.read_csv(self.sensor_dist_path)

        if os.path.exists(self.drone_info_path):
            print("  * Loading drone_information.csv...")
            self.df_drone_info = pd.read_csv(self.drone_info_path)

        if os.path.exists(self.drone_traj_path):
            print("  * Loading drone_trajectories.csv...")
            self.df_drone_traj = pd.read_csv(self.drone_traj_path)

        if os.path.exists(self.sensor_det_path):
            print("  * Loading sensor_detection_summary.csv...")
            self.df_sensor_det = pd.read_csv(self.sensor_det_path)

        print(f"[OK] All datasets loaded in {time.time() - t0:.2f} seconds.\n")

    def load_experiment(self, exp_id):
        """Filter cached datasets for a specific experiment ID."""
        self.exp_id = exp_id

        # 1. Sensors & Positions
        if self.df_sensor_dist is not None:
            exp_dist = self.df_sensor_dist[self.df_sensor_dist['exp_id'] == exp_id]
            if not exp_dist.empty:
                s1 = exp_dist[['sensor1_id', 'x1', 'y1', 'z1']].rename(
                    columns={'sensor1_id': 'sensor_id', 'x1': 'x', 'y1': 'y', 'z1': 'z'}
                )
                s2 = exp_dist[['sensor2_id', 'x2', 'y2', 'z2']].rename(
                    columns={'sensor2_id': 'sensor_id', 'x2': 'x', 'y2': 'y', 'z2': 'z'}
                )
                sensors_df = pd.concat([s1, s2]).drop_duplicates(subset=['sensor_id']).sort_values('sensor_id')

                if self.df_sensor_info is not None:
                    sensors_df = pd.merge(sensors_df, self.df_sensor_info, on='sensor_id', how='left')
                self.current_exp_sensors = sensors_df
            else:
                self.current_exp_sensors = pd.DataFrame()
        else:
            self.current_exp_sensors = pd.DataFrame()

        # 2. Drone Info
        if self.df_drone_info is not None:
            drone_row = self.df_drone_info[self.df_drone_info['experiment_id'] == exp_id]
            if not drone_row.empty:
                self.current_exp_drone_info = drone_row.iloc[0].to_dict()
            else:
                self.current_exp_drone_info = {}
        else:
            self.current_exp_drone_info = {}

        # 3. Drone Trajectory
        if self.df_drone_traj is not None:
            self.current_exp_traj = self.df_drone_traj[self.df_drone_traj['experiment_id'] == exp_id].reset_index(drop=True)
        else:
            self.current_exp_traj = pd.DataFrame()

        # 4. Sensor Detections
        if self.df_sensor_det is not None:
            self.current_exp_det = self.df_sensor_det[self.df_sensor_det['experiment_id'] == exp_id].reset_index(drop=True)
        else:
            self.current_exp_det = pd.DataFrame()

        self.current_step = 0

    def run_gui(self):
        """Initialize Matplotlib GUI layout."""
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(13, 8.5))
        self.fig.patch.set_facecolor('#1e1e2e')
        self.ax.set_facecolor('#181825')
        self.fig.canvas.manager.set_window_title(f"UAV & Sensor Visualization - Exp #{self.exp_id}")

        plt.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.22)

        # Global Z min/max for color scale
        z_values = []
        if self.df_sensor_dist is not None and not self.df_sensor_dist.empty:
            z_values.extend(self.df_sensor_dist['z1'].values)
            z_values.extend(self.df_sensor_dist['z2'].values)
        if self.df_drone_traj is not None and not self.df_drone_traj.empty:
            z_values.extend(self.df_drone_traj['start_z'].values)

        z_min = float(min(z_values)) if z_values else 0.0
        z_max = float(max(z_values)) if z_values else 100.0
        if z_min == z_max:
            z_max = z_min + 10.0

        self.norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
        self.cmap = cm.viridis
        self.sm = cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        self.sm.set_array([])

        cbar_ax = self.fig.add_axes([0.90, 0.22, 0.02, 0.68])
        self.cbar = self.fig.colorbar(self.sm, cax=cbar_ax)
        self.cbar.set_label("Z Axis Altitude (meters)", color="white", fontsize=11, fontweight='bold')
        self.cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(self.cbar.ax, 'yticklabels'), color="white")

        # Controls Setup
        num_steps = len(self.current_exp_traj) if not self.current_exp_traj.empty else 1

        # Timestep Slider
        ax_slider = plt.axes([0.15, 0.12, 0.55, 0.03], facecolor='#313244')
        self.slider = Slider(
            ax=ax_slider,
            label='Timestep ',
            valmin=0,
            valmax=max(0, num_steps - 1),
            valinit=0,
            valfmt='%d',
            color='#89b4fa'
        )
        self.slider.label.set_color('white')
        self.slider.valtext.set_color('white')
        self.slider.on_changed(self.on_slider_change)

        # Prev Button
        ax_prev = plt.axes([0.15, 0.04, 0.08, 0.04])
        self.btn_prev = Button(ax_prev, '< Prev', color='#313244', hovercolor='#45475a')
        self.btn_prev.label.set_color('white')
        self.btn_prev.on_clicked(self.prev_step)

        # Next Button
        ax_next = plt.axes([0.24, 0.04, 0.08, 0.04])
        self.btn_next = Button(ax_next, 'Next >', color='#313244', hovercolor='#45475a')
        self.btn_next.label.set_color('white')
        self.btn_next.on_clicked(self.next_step)

        # Play/Pause Button
        ax_play = plt.axes([0.33, 0.04, 0.10, 0.04])
        self.btn_play = Button(ax_play, 'Play', color='#a6e3a1', hovercolor='#94e2d5')
        self.btn_play.label.set_color('#11111b')
        self.btn_play.label.set_weight('bold')
        self.btn_play.on_clicked(self.toggle_play)

        # Exp ID TextBox
        ax_exp_box = plt.axes([0.55, 0.04, 0.15, 0.04], facecolor='#313244')
        self.txt_exp = TextBox(ax_exp_box, 'Exp ID: ', initial=str(self.exp_id), color='#313244', hovercolor='#45475a')
        self.txt_exp.label.set_color('white')
        self.txt_exp.text_disp.set_color('white')
        self.txt_exp.on_submit(self.on_exp_submit)

        # Mouse hover callback
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)

        # Initial Render
        self.render_plot()
        plt.show()

    def render_plot(self):
        """Re-draw plot axes, sensor points, UAV position, and trajectory trail."""
        self.ax.clear()
        self.ax.grid(True, linestyle='--', alpha=0.3, color='#585b70')
        self.ax.set_xlabel("X Position (m)", fontsize=11, color='white')
        self.ax.set_ylabel("Y Position (m)", fontsize=11, color='white')

        # Tooltip annotation object
        self.annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.6", fc="#1e1e2e", ec="#cba6f7", lw=1.8, alpha=0.95),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="#cba6f7", lw=1.8),
            color="white",
            fontsize=9,
            family="monospace"
        )
        self.annot.set_visible(False)

        # 1. Trajectory Trail
        if self.current_exp_traj is not None and not self.current_exp_traj.empty:
            traj_x = self.current_exp_traj['start_x'].values
            traj_y = self.current_exp_traj['start_y'].values
            self.ax.plot(traj_x, traj_y, color='#89b4fa', linestyle=':', linewidth=1.8, alpha=0.6, label='UAV Trajectory Trail')

        # 2. Sensors
        self.sensor_data_list = []
        if self.current_exp_sensors is not None and not self.current_exp_sensors.empty:
            for idx, srow in self.current_exp_sensors.iterrows():
                sx, sy, sz = srow['x'], srow['y'], srow['z']
                sensor_id = int(srow['sensor_id'])
                s_color = self.cmap(self.norm(sz))

                # Sensor circle marker
                self.ax.scatter(
                    sx, sy, c=[s_color], s=180, marker='o', edgecolors='white', linewidths=1.5, zorder=3
                )
                self.ax.text(sx, sy + 35, f"S{sensor_id}", color='white', fontsize=9, ha='center', fontweight='bold')
                self.sensor_data_list.append(srow.to_dict())

        # 3. Current UAV Position
        self.uav_data = None
        if self.current_exp_traj is not None and not self.current_exp_traj.empty:
            step = min(self.current_step, len(self.current_exp_traj) - 1)
            urow = self.current_exp_traj.iloc[step].to_dict()

            ux, uy, uz = urow['start_x'], urow['start_y'], urow['start_z']
            u_color = self.cmap(self.norm(uz))

            # UAV Star Marker + Ring
            self.ax.scatter(ux, uy, c=[u_color], s=400, marker='*', edgecolors='#f38ba8', linewidths=2.0, zorder=5, label='UAV Position')
            self.ax.scatter(ux, uy, c='none', s=700, marker='o', edgecolors='#f38ba8', linewidths=1.5, linestyle='--', zorder=4)

            drone_type = self.current_exp_drone_info.get('drone_type', 'UAV')
            self.ax.text(ux, uy - 50, f"UAV ({drone_type})", color='#f38ba8', fontsize=10, ha='center', fontweight='bold')
            self.uav_data = urow

            timestamp = urow.get('timestamp', '')
            speed = urow.get('speed', 0.0)
            title_text = (f"Experiment #{self.exp_id} | Drone #{urow.get('drone_id', 1)} ({drone_type}) | "
                          f"Timestep {step}/{len(self.current_exp_traj)-1} | Speed: {speed:.1f} m/s")
            self.ax.set_title(title_text, fontsize=12, color='white', fontweight='bold')

        # Auto axis limits with padding
        all_x, all_y = [], []
        if self.current_exp_sensors is not None and not self.current_exp_sensors.empty:
            all_x.extend(self.current_exp_sensors['x'].values)
            all_y.extend(self.current_exp_sensors['y'].values)
        if self.current_exp_traj is not None and not self.current_exp_traj.empty:
            all_x.extend(self.current_exp_traj['start_x'].values)
            all_y.extend(self.current_exp_traj['start_y'].values)

        if all_x and all_y:
            pad_x = (max(all_x) - min(all_x)) * 0.1 + 100
            pad_y = (max(all_y) - min(all_y)) * 0.1 + 100
            self.ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
            self.ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)

        self.ax.legend(loc='upper right', facecolor='#313244', edgecolor='#585b70', labelcolor='white')
        self.fig.canvas.draw_idle()

    def update_tooltip_sensor(self, sdata):
        """Build multi-line string for Sensor hover tooltip."""
        sid = int(sdata.get('sensor_id', 0))
        stype = sdata.get('sensor_type', 'RF')
        x, y, z = sdata.get('x', 0.0), sdata.get('y', 0.0), sdata.get('z', 0.0)
        sens = sdata.get('sensitivity', 0.0)
        sinr = sdata.get('sinr_threshold', 0.0)
        gain = sdata.get('gain', 0.0)
        fmin = sdata.get('freq_min', 0.0)
        fmax = sdata.get('freq_max', 0.0)
        sectors = sdata.get('sector_count', 'N/A')
        clock_off = sdata.get('clock_offset', 0.0)

        # Check current timestep detection info
        det_info = "No detection at current timestep"
        if self.current_exp_det is not None and not self.current_exp_det.empty and self.current_exp_traj is not None:
            cur_ts = self.current_exp_traj.iloc[self.current_step].get('timestamp', None)
            if cur_ts:
                matched = self.current_exp_det[
                    (self.current_exp_det['sensor_id'] == sid) & (self.current_exp_det['timestamp'] == cur_ts)
                ]
                if not matched.empty:
                    row = matched.iloc[0]
                    pwr = row.get('total_power_dBm', 'N/A')
                    dets = row.get('num_detections', 'N/A')
                    sec_det = row.get('sensor_sectors_detected', 'N/A')
                    det_info = f"Power: {pwr} dBm | Detections: {dets}\n  Sectors: {sec_det}"

        text = (
            f"SENSOR #{sid} ({stype})\n"
            f"-----------------------------\n"
            f"Position   : X={x:.1f}m, Y={y:.1f}m, Z={z:.1f}m\n"
            f"Sensitivity: {sens:.2f} dBm\n"
            f"SINR Thresh: {sinr:.2f} dB\n"
            f"Gain       : {gain:.2f} dBi\n"
            f"Frequency  : {fmin/1e6:.1f} - {fmax/1e6:.1f} MHz\n"
            f"Sectors    : {sectors}\n"
            f"Clock Offset: {clock_off}\n"
            f"-----------------------------\n"
            f"Step Status:\n  {det_info}"
        )
        return text

    def update_tooltip_uav(self, udata):
        """Build multi-line string for UAV hover tooltip."""
        did = int(udata.get('drone_id', 1))
        dtype = self.current_exp_drone_info.get('drone_type', 'UAV')
        power = self.current_exp_drone_info.get('drone_transmitter_power', 0.0)
        tx_gain = self.current_exp_drone_info.get('transmitter_antenna_gain', 0.0)
        bw = self.current_exp_drone_info.get('bandwidth', 0.0)
        fmin = self.current_exp_drone_info.get('freq_min', 0.0)
        fmax = self.current_exp_drone_info.get('freq_max', 0.0)

        sx, sy, sz = udata.get('start_x', 0.0), udata.get('start_y', 0.0), udata.get('start_z', 0.0)
        ex, ey, ez = udata.get('end_x', 0.0), udata.get('end_y', 0.0), udata.get('end_z', 0.0)
        speed = udata.get('speed', 0.0)
        ts = udata.get('timestamp', '')

        yaw_chg = bool(udata.get('is_yaw_change', 0))
        pitch_chg = bool(udata.get('is_pitch_change', 0))
        spd_chg = bool(udata.get('is_speed_change', 0))
        anomaly = bool(udata.get('is_anomaly', 0))

        text = (
            f"UAV #{did} ({dtype})\n"
            f"-----------------------------\n"
            f"Position   : X={sx:.1f}m, Y={sy:.1f}m, Z={sz:.1f}m\n"
            f"Target Pos : X={ex:.1f}m, Y={ey:.1f}m, Z={ez:.1f}m\n"
            f"Speed      : {speed:.2f} m/s\n"
            f"Timestamp  : {ts}\n"
            f"-----------------------------\n"
            f"Tx Power   : {power:.2f} dBm | Gain: {tx_gain:.1f} dBi\n"
            f"Bandwidth  : {bw/1e6:.1f} MHz ({fmin/1e6:.1f}-{fmax/1e6:.1f}MHz)\n"
            f"Flags      : Yaw={yaw_chg}, Pitch={pitch_chg}, Speed={spd_chg}\n"
            f"Anomaly    : {anomaly}"
        )
        return text

    def on_hover(self, event):
        """Detect mouse hover in pixel coordinates for precision."""
        if event.inaxes != self.ax:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Transform mouse and objects to display pixels
        trans = self.ax.transData.transform

        hover_found = False

        # Check Sensors proximity (within 25 pixels)
        if self.sensor_data_list:
            for sdata in self.sensor_data_list:
                sx, sy = sdata['x'], sdata['y']
                px_m, py_m = trans((x, y))
                px_s, py_s = trans((sx, sy))
                dist_px = np.hypot(px_m - px_s, py_m - py_s)

                if dist_px < 25.0:
                    self.annot.xy = (sx, sy)
                    self.annot.set_text(self.update_tooltip_sensor(sdata))
                    self.annot.get_bbox_patch().set_facecolor("#1e1e2e")
                    self.annot.get_bbox_patch().set_edgecolor("#cba6f7")
                    self.annot.set_visible(True)
                    self.fig.canvas.draw_idle()
                    hover_found = True
                    break

        # Check UAV proximity (within 30 pixels)
        if not hover_found and self.uav_data is not None:
            ux, uy = self.uav_data['start_x'], self.uav_data['start_y']
            px_m, py_m = trans((x, y))
            px_u, py_u = trans((ux, uy))
            dist_px = np.hypot(px_m - px_u, py_m - py_u)

            if dist_px < 30.0:
                self.annot.xy = (ux, uy)
                self.annot.set_text(self.update_tooltip_uav(self.uav_data))
                self.annot.get_bbox_patch().set_facecolor("#11111b")
                self.annot.get_bbox_patch().set_edgecolor("#f38ba8")
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
                hover_found = True

        if not hover_found and self.annot.get_visible():
            self.annot.set_visible(False)
            self.fig.canvas.draw_idle()

    def on_slider_change(self, val):
        self.current_step = int(val)
        self.render_plot()

    def prev_step(self, event=None):
        if self.current_step > 0:
            self.current_step -= 1
            self.slider.set_val(self.current_step)

    def next_step(self, event=None):
        max_step = len(self.current_exp_traj) - 1 if self.current_exp_traj is not None else 0
        if self.current_step < max_step:
            self.current_step += 1
            self.slider.set_val(self.current_step)
        else:
            if self.is_playing:
                self.toggle_play()

    def toggle_play(self, event=None):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.label.set_text('Pause')
            self.btn_play.color = '#f38ba8'
            self.btn_play.hovercolor = '#eba0ac'
            self.start_timer()
        else:
            self.btn_play.label.set_text('Play')
            self.btn_play.color = '#a6e3a1'
            self.btn_play.hovercolor = '#94e2d5'
            self.stop_timer()

    def start_timer(self):
        if self.is_playing:
            self.timer = self.fig.canvas.new_timer(interval=400)
            self.timer.add_callback(self.on_timer_tick)
            self.timer.start()

    def on_timer_tick(self):
        if self.is_playing:
            max_step = len(self.current_exp_traj) - 1 if self.current_exp_traj is not None else 0
            if self.current_step >= max_step:
                self.current_step = 0
            else:
                self.current_step += 1
            self.slider.set_val(self.current_step)

    def stop_timer(self):
        if self.timer:
            self.timer.stop()
            self.timer = None

    def on_exp_submit(self, text):
        try:
            new_id = int(text)
            self.load_experiment(new_id)
            num_steps = len(self.current_exp_traj) if not self.current_exp_traj.empty else 1
            self.slider.valmax = max(0, num_steps - 1)
            self.slider.set_val(0)
            self.render_plot()
        except ValueError:
            print(f"Invalid Experiment ID input: {text}")

    def export_html(self, output_file="experiment_visualization.html"):
        """Generate an interactive HTML visualization page using Plotly.js."""
        print(f"Exporting interactive HTML visualization to {output_file}...")

        sensors = []
        if self.current_exp_sensors is not None:
            for _, r in self.current_exp_sensors.iterrows():
                sensors.append({
                    "id": int(r.get('sensor_id', 0)),
                    "type": str(r.get('sensor_type', 'RF')),
                    "x": float(r.get('x', 0)),
                    "y": float(r.get('y', 0)),
                    "z": float(r.get('z', 0)),
                    "sensitivity": float(r.get('sensitivity', 0)),
                    "sinr_threshold": float(r.get('sinr_threshold', 0)),
                    "gain": float(r.get('gain', 0)),
                    "freq_min": float(r.get('freq_min', 0)),
                    "freq_max": float(r.get('freq_max', 0)),
                    "sector_count": int(r.get('sector_count', 0)) if pd.notnull(r.get('sector_count')) else 0
                })

        trajectory = []
        if self.current_exp_traj is not None:
            for _, r in self.current_exp_traj.iterrows():
                trajectory.append({
                    "step": int(r.name),
                    "timestamp": str(r.get('timestamp', '')),
                    "speed": float(r.get('speed', 0)),
                    "x": float(r.get('start_x', 0)),
                    "y": float(r.get('start_y', 0)),
                    "z": float(r.get('start_z', 0)),
                    "end_x": float(r.get('end_x', 0)),
                    "end_y": float(r.get('end_y', 0)),
                    "end_z": float(r.get('end_z', 0)),
                    "is_yaw_change": int(r.get('is_yaw_change', 0)),
                    "is_pitch_change": int(r.get('is_pitch_change', 0)),
                    "is_speed_change": int(r.get('is_speed_change', 0)),
                    "is_anomaly": int(r.get('is_anomaly', 0))
                })

        drone_info = {
            "type": str(self.current_exp_drone_info.get('drone_type', 'UAV')),
            "power": float(self.current_exp_drone_info.get('drone_transmitter_power', 0)),
            "gain": float(self.current_exp_drone_info.get('transmitter_antenna_gain', 0)),
            "bw": float(self.current_exp_drone_info.get('bandwidth', 0)),
            "freq_min": float(self.current_exp_drone_info.get('freq_min', 0)),
            "freq_max": float(self.current_exp_drone_info.get('freq_max', 0))
        }

        # Build Hover Texts in Python to avoid JS string formatting issues
        sensor_hover_texts = [
            f"<b>SENSOR #{s['id']} ({s['type']})</b><br>"
            f"Pos: X={s['x']:.1f}m, Y={s['y']:.1f}m, Z={s['z']:.1f}m<br>"
            f"Sensitivity: {s['sensitivity']:.2f} dBm<br>"
            f"SINR Thresh: {s['sinr_threshold']:.2f} dB<br>"
            f"Gain: {s['gain']:.2f} dBi<br>"
            f"Freq: {s['freq_min']/1e6:.1f} - {s['freq_max']/1e6:.1f} MHz"
            for s in sensors
        ]

        uav_hover_texts = [
            f"<b>UAV ({drone_info['type']})</b><br>"
            f"Step: {t['step']}<br>"
            f"Pos: X={t['x']:.1f}m, Y={t['y']:.1f}m, Z={t['z']:.1f}m<br>"
            f"Speed: {t['speed']:.2f} m/s<br>"
            f"Time: {t['timestamp']}<br>"
            f"Tx Power: {drone_info['power']:.2f} dBm<br>"
            f"Flags: Yaw={t['is_yaw_change']}, Pitch={t['is_pitch_change']}, Spd={t['is_speed_change']}<br>"
            f"Anomaly: {t['is_anomaly']}"
            for t in trajectory
        ]

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>UAV & Multi-Sensor Experiment #{self.exp_id} Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            background-color: #1e1e2e;
            color: #cdd6f4;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        h2 {{
            color: #89b4fa;
            margin-bottom: 5px;
        }}
        #plot {{
            width: 100%;
            height: 75vh;
            background-color: #181825;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        .controls {{
            margin-top: 15px;
            display: flex;
            align-items: center;
            gap: 15px;
            background: #313244;
            padding: 15px;
            border-radius: 10px;
        }}
        button {{
            background-color: #89b4fa;
            color: #11111b;
            border: none;
            padding: 10px 18px;
            font-weight: bold;
            border-radius: 6px;
            cursor: pointer;
        }}
        button:hover {{ background-color: #b4befe; }}
        input[type=range] {{ flex-grow: 1; }}
    </style>
</head>
<body>
    <h2>UAV & Multi-Sensor Interactive Experiment #{self.exp_id}</h2>
    <div id="plot"></div>
    <div class="controls">
        <button id="btnPrev">Prev Step</button>
        <button id="btnPlay">Play</button>
        <button id="btnNext">Next Step</button>
        <span>Timestep: <strong id="stepVal">0</strong> / {len(trajectory)-1}</span>
        <input type="range" id="slider" min="0" max="{len(trajectory)-1}" value="0" />
    </div>

    <script>
        const sensors = {json.dumps(sensors)};
        const trajectory = {json.dumps(trajectory)};
        const sensorHoverTexts = {json.dumps(sensor_hover_texts)};
        const uavHoverTexts = {json.dumps(uav_hover_texts)};

        let currentStep = 0;
        let isPlaying = false;
        let playTimer = null;

        function updatePlot() {{
            const curUav = trajectory[currentStep] || {{x: 0, y: 0, z: 0}};

            const sensorTrace = {{
                x: sensors.map(s => s.x),
                y: sensors.map(s => s.y),
                text: sensorHoverTexts,
                hoverinfo: 'text',
                mode: 'markers+text',
                marker: {{
                    size: 16,
                    color: sensors.map(s => s.z),
                    colorscale: 'Viridis',
                    colorbar: {{ title: 'Z Altitude (m)' }},
                    line: {{ color: 'white', width: 1.5 }}
                }},
                textposition: 'top center',
                name: 'Sensors'
            }};

            const trajTrace = {{
                x: trajectory.map(t => t.x),
                y: trajectory.map(t => t.y),
                mode: 'lines',
                line: {{ color: '#89b4fa', width: 2, dash: 'dot' }},
                name: 'Trajectory Trail',
                hoverinfo: 'none'
            }};

            const uavTrace = {{
                x: [curUav.x],
                y: [curUav.y],
                text: [uavHoverTexts[currentStep] || 'UAV'],
                hoverinfo: 'text',
                mode: 'markers',
                marker: {{
                    size: 22,
                    symbol: 'star',
                    color: [curUav.z],
                    colorscale: 'Viridis',
                    cmin: Math.min(...sensors.map(s => s.z).concat(trajectory.map(t => t.z))),
                    cmax: Math.max(...sensors.map(s => s.z).concat(trajectory.map(t => t.z))),
                    line: {{ color: '#f38ba8', width: 2.5 }}
                }},
                name: 'UAV Position'
            }};

            const layout = {{
                paper_bgcolor: '#1e1e2e',
                plot_bgcolor: '#181825',
                title: {{ text: `Experiment #${self.exp_id} | Timestep ${{currentStep}}`, font: {{ color: '#cdd6f4', size: 16 }} }},
                xaxis: {{ title: 'X Position (m)', color: '#cdd6f4', gridcolor: '#313244' }},
                yaxis: {{ title: 'Y Position (m)', color: '#cdd6f4', gridcolor: '#313244' }},
                legend: {{ font: {{ color: '#cdd6f4' }} }}
            }};

            Plotly.react('plot', [sensorTrace, trajTrace, uavTrace], layout);
            document.getElementById('stepVal').innerText = currentStep;
            document.getElementById('slider').value = currentStep;
        }}

        document.getElementById('slider').addEventListener('input', (e) => {{
            currentStep = parseInt(e.target.value);
            updatePlot();
        }});

        document.getElementById('btnPrev').addEventListener('click', () => {{
            if (currentStep > 0) {{ currentStep--; updatePlot(); }}
        }});

        document.getElementById('btnNext').addEventListener('click', () => {{
            if (currentStep < trajectory.length - 1) {{ currentStep++; updatePlot(); }}
        }});

        document.getElementById('btnPlay').addEventListener('click', () => {{
            isPlaying = !isPlaying;
            document.getElementById('btnPlay').innerText = isPlaying ? 'Pause' : 'Play';
            if (isPlaying) {{
                playTimer = setInterval(() => {{
                    if (currentStep < trajectory.length - 1) {{ currentStep++; }} else {{ currentStep = 0; }}
                    updatePlot();
                }}, 400);
            }} else {{
                clearInterval(playTimer);
            }}
        }});

        updatePlot();
    </script>
</body>
</html>
"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[OK] Interactive HTML saved to: {os.path.abspath(output_file)}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    default_data_dir = os.path.join(root_dir, "Data", "test_data_10_sensors_zsectors_ptx_model_preprocessing", "5K_4")
    parser = argparse.ArgumentParser(description="UAV & Sensor Experiment Visualizer")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_data_dir,
        help="Path to directory containing experiment CSV files"
    )
    parser.add_argument("--exp_id", type=int, default=1, help="Experiment ID to visualize")
    parser.add_argument("--html", action="store_true", help="Export as interactive HTML file")
    parser.add_argument("--output_html", type=str, default="experiment_visualization.html", help="HTML export filename")
    args = parser.parse_args()

    visualizer = ExperimentVisualizer(data_dir=args.data_dir, exp_id=args.exp_id)

    if args.html:
        visualizer.export_html(args.output_html)
    else:
        visualizer.run_gui()


if __name__ == "__main__":
    main()
