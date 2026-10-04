import os
import sys
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

        # Data placeholders
        self.df_sensor_info = None
        self.df_sensor_dist = None
        self.df_drone_info = None
        self.df_drone_traj = None
        self.df_sensor_det = None

        self.current_exp_sensors = None
        self.current_exp_drone_info = None
        self.current_exp_traj = None
        self.current_exp_det = None

        self.current_step = 0
        self.is_playing = False
        self.timer = None

        self.load_metadata()
        self.load_experiment(exp_id)
        self.setup_gui()

    def load_metadata(self):
        """Load static sensor and drone metadata tables."""
        print(f"Loading metadata from {self.data_dir}...")
        if os.path.exists(self.sensor_info_path):
            self.df_sensor_info = pd.read_csv(self.sensor_info_path)
            # Set index for fast lookup if sensor_id present
            if "sensor_id" in self.df_sensor_info.columns:
                self.df_sensor_info.set_index("sensor_id", inplace=False)

        if os.path.exists(self.drone_info_path):
            self.df_drone_info = pd.read_csv(self.drone_info_path)

    def load_experiment(self, exp_id):
        """Load data for a specific experiment ID."""
        self.exp_id = exp_id
        print(f"Loading Experiment ID {exp_id}...")

        # 1. Load Sensors for this experiment from sensor_distances.csv
        # Each row has (exp_id, sensor1_id, x1, y1, z1, sensor2_id, x2, y2, z2, d)
        if os.path.exists(self.sensor_dist_path):
            # Load chunk/filtered for exp_id
            df_dist = pd.read_csv(self.sensor_dist_path)
            exp_dist = df_dist[df_dist['exp_id'] == exp_id]
            if not exp_dist.empty:
                s1 = exp_dist[['sensor1_id', 'x1', 'y1', 'z1']].rename(
                    columns={'sensor1_id': 'sensor_id', 'x1': 'x', 'y1': 'y', 'z1': 'z'}
                )
                s2 = exp_dist[['sensor2_id', 'x2', 'y2', 'z2']].rename(
                    columns={'sensor2_id': 'sensor_id', 'x2': 'x', 'y2': 'y', 'z2': 'z'}
                )
                sensors_df = pd.concat([s1, s2]).drop_duplicates(subset=['sensor_id']).sort_values('sensor_id')

                # Merge with sensor_information details if available
                if self.df_sensor_info is not None:
                    sensors_df = pd.merge(sensors_df, self.df_sensor_info, on='sensor_id', how='left')
                self.current_exp_sensors = sensors_df
            else:
                self.current_exp_sensors = pd.DataFrame()
        else:
            self.current_exp_sensors = pd.DataFrame()

        # 2. Drone Information for this experiment
        if self.df_drone_info is not None:
            drone_row = self.df_drone_info[self.df_drone_info['experiment_id'] == exp_id]
            if not drone_row.empty:
                self.current_exp_drone_info = drone_row.iloc[0].to_dict()
            else:
                self.current_exp_drone_info = {}

        # 3. Drone Trajectory for this experiment
        if os.path.exists(self.drone_traj_path):
            # Filtering trajectory for experiment_id
            df_traj_all = pd.read_csv(self.drone_traj_path)
            self.current_exp_traj = df_traj_all[df_traj_all['experiment_id'] == exp_id].reset_index(drop=True)
        else:
            self.current_exp_traj = pd.DataFrame()

        # 4. Sensor Detections for this experiment
        if os.path.exists(self.sensor_det_path):
            df_det_all = pd.read_csv(self.sensor_det_path)
            self.current_exp_det = df_det_all[df_det_all['experiment_id'] == exp_id].reset_index(drop=True)
        else:
            self.current_exp_det = pd.DataFrame()

        self.current_step = 0

    def setup_gui(self):
        """Set up Matplotlib figure, canvas, widgets and hover listeners."""
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.patch.set_facecolor('#1e1e2e')
        self.ax.set_facecolor('#181825')

        # Adjust subplot axes to leave space for slider and controls at the bottom
        plt.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.22)

        # Plot setup
        self.ax.set_title(f"Experiment {self.exp_id} - UAV & Multi-Sensor Visualization", fontsize=14, color='white', fontweight='bold')
        self.ax.set_xlabel("X Position (m)", fontsize=11, color='white')
        self.ax.set_ylabel("Y Position (m)", fontsize=11, color='white')
        self.ax.grid(True, linestyle='--', alpha=0.3, color='#585b70')

        # Hover Annotation box
        self.annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.6", fc="#313244", ec="#cba6f7", lw=1.5, alpha=0.95),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="#cba6f7", lw=1.5),
            color="white",
            fontsize=9,
            family="monospace"
        )
        self.annot.set_visible(False)

        # Colorbar setup for Z height
        # Find global Z min/max for color scale normalization
        z_values = []
        if not self.current_exp_sensors.empty and 'z' in self.current_exp_sensors.columns:
            z_values.extend(self.current_exp_sensors['z'].values)
        if not self.current_exp_traj.empty:
            z_values.extend(self.current_exp_traj['start_z'].values)

        z_min = min(z_values) if z_values else 0
        z_max = max(z_values) if z_values else 100
        if z_min == z_max:
            z_max = z_min + 10.0

        self.norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
        self.cmap = cm.viridis
        self.sm = cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        self.sm.set_array([])

        cbar_ax = self.fig.add_axes([0.90, 0.22, 0.02, 0.70])
        self.cbar = self.fig.colorbar(self.sm, cax=cbar_ax)
        self.cbar.set_label("Z Altitude (m)", color="white", fontsize=11)
        self.cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(self.cbar.ax, 'yticklabels'), color="white")

        # Interactive Widgets (Slider, Buttons, Text Box)
        # Slider for Timesteps
        num_steps = len(self.current_exp_traj) if not self.current_exp_traj.empty else 1
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
        ax_prev = plt.axes([0.15, 0.05, 0.08, 0.04])
        self.btn_prev = Button(ax_prev, '◄ Prev', color='#313244', hovercolor='#45475a')
        self.btn_prev.label.set_color('white')
        self.btn_prev.on_clicked(self.prev_step)

        # Next Button
        ax_next = plt.axes([0.24, 0.05, 0.08, 0.04])
        self.btn_next = Button(ax_next, 'Next ►', color='#313244', hovercolor='#45475a')
        self.btn_next.label.set_color('white')
        self.btn_next.on_clicked(self.next_step)

        # Play/Pause Button
        ax_play = plt.axes([0.33, 0.05, 0.10, 0.04])
        self.btn_play = Button(ax_play, '▶ Play', color='#a6e3a1', hovercolor='#94e2d5')
        self.btn_play.label.set_color('#11111b')
        self.btn_play.label.set_weight('bold')
        self.btn_play.on_clicked(self.toggle_play)

        # Exp ID TextBox
        ax_exp_box = plt.axes([0.55, 0.05, 0.15, 0.04], facecolor='#313244')
        self.txt_exp = TextBox(ax_exp_box, 'Exp ID: ', initial=str(self.exp_id), color='#313244', hovercolor='#45475a')
        self.txt_exp.label.set_color('white')
        self.txt_exp.text_disp.set_color('white')
        self.txt_exp.on_submit(self.on_exp_submit)

        # Mouse hover connection
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)

        # Initial Plot Render
        self.render_plot()

    def render_plot(self):
        """Draw current state of sensors, trajectory, and UAV position."""
        self.ax.clear()
        self.ax.grid(True, linestyle='--', alpha=0.3, color='#585b70')
        self.ax.set_xlabel("X Position (m)", fontsize=11, color='white')
        self.ax.set_ylabel("Y Position (m)", fontsize=11, color='white')

        # Re-attach annotation
        self.annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.6", fc="#313244", ec="#cba6f7", lw=1.5, alpha=0.95),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="#cba6f7", lw=1.5),
            color="white",
            fontsize=9,
            family="monospace"
        )
        self.annot.set_visible(False)

        # 1. Plot Trajectory Trail
        if self.current_exp_traj is not None and not self.current_exp_traj.empty:
            traj_x = self.current_exp_traj['start_x'].values
            traj_y = self.current_exp_traj['start_y'].values
            self.ax.plot(traj_x, traj_y, color='#89b4fa', linestyle=':', linewidth=1.5, alpha=0.6, label='UAV Trajectory Trail')

        # 2. Plot Sensors
        self.sensor_scatters = []
        self.sensor_data_list = []
        if self.current_exp_sensors is not None and not self.current_exp_sensors.empty:
            for idx, srow in self.current_exp_sensors.iterrows():
                sx, sy, sz = srow['x'], srow['y'], srow['z']
                sensor_id = srow['sensor_id']
                s_color = self.cmap(self.norm(sz))

                # Plot sensor marker (Circle with dark border)
                sc = self.ax.scatter(
                    sx, sy, c=[s_color], s=160, marker='o', edgecolors='white', linewidths=1.5, zorder=3
                )
                self.ax.text(sx, sy + 30, f"S{int(sensor_id)}", color='white', fontsize=9, ha='center', fontweight='bold')
                self.sensor_scatters.append(sc)
                self.sensor_data_list.append(srow.to_dict())

        # 3. Plot Current UAV Position
        self.uav_scatter = None
        self.uav_data = None
        if self.current_exp_traj is not None and not self.current_exp_traj.empty:
            step = min(self.current_step, len(self.current_exp_traj) - 1)
            urow = self.current_exp_traj.iloc[step].to_dict()

            ux, uy, uz = urow['start_x'], urow['start_y'], urow['start_z']
            u_color = self.cmap(self.norm(uz))

            # UAV Marker (Star / Diamond with glowing aura)
            self.ax.scatter(ux, uy, c=[u_color], s=350, marker='*', edgecolors='#f38ba8', linewidths=2.0, zorder=5, label='UAV Position')
            self.ax.scatter(ux, uy, c='none', s=600, marker='o', edgecolors='#f38ba8', linewidths=1.2, linestyle='--', zorder=4)

            drone_type = self.current_exp_drone_info.get('drone_type', 'UAV')
            self.ax.text(ux, uy - 45, f"{drone_type} (Step {step})", color='#f38ba8', fontsize=10, ha='center', fontweight='bold')
            self.uav_data = urow

            timestamp = urow.get('timestamp', '')
            speed = urow.get('speed', 0.0)
            title_text = (f"Exp {self.exp_id} | Drone #{urow.get('drone_id', 1)} ({drone_type}) | "
                          f"Step {step}/{len(self.current_exp_traj)-1} | Speed: {speed:.1f} m/s | Time: {timestamp}")
            self.ax.set_title(title_text, fontsize=12, color='white', fontweight='bold')

        self.ax.legend(loc='upper right', facecolor='#313244', edgecolor='#585b70', labelcolor='white')
        self.fig.canvas.draw_idle()

    def update_tooltip_sensor(self, sdata):
        """Format detailed string for Sensor hover tooltip."""
        sid = int(sdata.get('sensor_id', 0))
        stype = sdata.get('sensor_type', 'RF')
        x, y, z = sdata.get('x', 0.0), sdata.get('y', 0.0), sdata.get('z', 0.0)
        sens = sdata.get('sensitivity', 'N/A')
        sinr = sdata.get('sinr_threshold', 'N/A')
        gain = sdata.get('gain', 'N/A')
        fmin = sdata.get('freq_min', 'N/A')
        fmax = sdata.get('freq_max', 'N/A')
        sectors = sdata.get('sector_count', 'N/A')
        clock_off = sdata.get('clock_offset', 'N/A')

        # Check current timestep detection info
        det_info = "No detection at step"
        if self.current_exp_det is not None and not self.current_exp_det.empty and self.current_exp_traj is not None:
            cur_ts = self.current_exp_traj.iloc[self.current_step].get('timestamp', None)
            if cur_ts:
                matched = self.current_exp_det[
                    (self.current_exp_det['sensor_id'] == sid) & (self.current_exp_det['timestamp'] == cur_ts)
                ]
                if not matched.empty:
                    pwr = matched.iloc[0].get('total_power_dBm', 'N/A')
                    dets = matched.iloc[0].get('num_detections', 'N/A')
                    sec_det = matched.iloc[0].get('sensor_sectors_detected', 'N/A')
                    det_info = f"Pwr: {pwr} dBm | Dets: {dets} | Sec: {sec_det}"

        text = (
            f"📡 SENSOR #{sid} ({stype})\n"
            f"---------------------------\n"
            f"Position: X={x:.1f}m, Y={y:.1f}m, Z={z:.1f}m\n"
            f"Sensitivity: {sens:.2f} dBm\n"
            f"SINR Thresh: {sinr:.2f} dB\n"
            f"Gain: {gain:.2f} dBi\n"
            f"Frequency: {fmin/1e6:.1f} - {fmax/1e6:.1f} MHz\n"
            f"Sectors: {sectors} | Clock Offset: {clock_off}\n"
            f"---------------------------\n"
            f"Step Status: {det_info}"
        )
        return text

    def update_tooltip_uav(self, udata):
        """Format detailed string for UAV hover tooltip."""
        did = int(udata.get('drone_id', 1))
        dtype = self.current_exp_drone_info.get('drone_type', 'UAV')
        power = self.current_exp_drone_info.get('drone_transmitter_power', 'N/A')
        tx_gain = self.current_exp_drone_info.get('transmitter_antenna_gain', 'N/A')
        bw = self.current_exp_drone_info.get('bandwidth', 'N/A')
        fmin = self.current_exp_drone_info.get('freq_min', 'N/A')
        fmax = self.current_exp_drone_info.get('freq_max', 'N/A')

        sx, sy, sz = udata.get('start_x', 0.0), udata.get('start_y', 0.0), udata.get('start_z', 0.0)
        ex, ey, ez = udata.get('end_x', 0.0), udata.get('end_y', 0.0), udata.get('end_z', 0.0)
        speed = udata.get('speed', 0.0)
        ts = udata.get('timestamp', '')

        yaw_chg = bool(udata.get('is_yaw_change', 0))
        pitch_chg = bool(udata.get('is_pitch_change', 0))
        spd_chg = bool(udata.get('is_speed_change', 0))
        anomaly = bool(udata.get('is_anomaly', 0))

        text = (
            f"🛸 UAV #{did} ({dtype})\n"
            f"---------------------------\n"
            f"Start Pos: X={sx:.1f}m, Y={sy:.1f}m, Z={sz:.1f}m\n"
            f"End Pos  : X={ex:.1f}m, Y={ey:.1f}m, Z={ez:.1f}m\n"
            f"Speed    : {speed:.2f} m/s\n"
            f"Timestamp: {ts}\n"
            f"---------------------------\n"
            f"Tx Power : {power:.2f} dBm | Tx Gain: {tx_gain} dBi\n"
            f"Bandwidth: {bw/1e6:.1f} MHz ({fmin/1e6:.1f}-{fmax/1e6:.1f}MHz)\n"
            f"Flags    : YawChg={yaw_chg}, PitchChg={pitch_chg}, SpdChg={spd_chg}\n"
            f"Anomaly  : {anomaly}"
        )
        return text

    def on_hover(self, event):
        """Callback when mouse moves over canvas to show tooltips."""
        if event.inaxes != self.ax:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Check proximity to sensors
        hover_found = False
        if self.sensor_data_list:
            for sdata in self.sensor_data_list:
                sx, sy = sdata['x'], sdata['y']
                dist = np.hypot(x - sx, y - sy)
                # Distance threshold for hover detection (in data units)
                if dist < 120.0:  # ~120 meters radius hover box
                    self.annot.xy = (sx, sy)
                    self.annot.set_text(self.update_tooltip_sensor(sdata))
                    self.annot.get_bbox_patch().set_facecolor("#313244")
                    self.annot.get_bbox_patch().set_edgecolor("#cba6f7")
                    self.annot.set_visible(True)
                    self.fig.canvas.draw_idle()
                    hover_found = True
                    break

        # Check proximity to UAV
        if not hover_found and self.uav_data is not None:
            ux, uy = self.uav_data['start_x'], self.uav_data['start_y']
            dist = np.hypot(x - ux, y - uy)
            if dist < 150.0:
                self.annot.xy = (ux, uy)
                self.annot.set_text(self.update_tooltip_uav(self.uav_data))
                self.annot.get_bbox_patch().set_facecolor("#1e1e2e")
                self.annot.get_bbox_patch().set_edgecolor("#f38ba8")
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
                hover_found = True

        if not hover_found and self.annot.get_visible():
            self.annot.set_visible(False)
            self.fig.canvas.draw_idle()

    def on_slider_change(self, val):
        """Handle timestep slider change."""
        self.current_step = int(val)
        self.render_plot()

    def prev_step(self, event=None):
        """Move to previous timestep."""
        if self.current_step > 0:
            self.current_step -= 1
            self.slider.set_val(self.current_step)

    def next_step(self, event=None):
        """Move to next timestep."""
        max_step = len(self.current_exp_traj) - 1 if self.current_exp_traj is not None else 0
        if self.current_step < max_step:
            self.current_step += 1
            self.slider.set_val(self.current_step)
        else:
            if self.is_playing:
                self.toggle_play()  # Stop play at end

    def toggle_play(self, event=None):
        """Start/stop automatic timestep playback."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.label.set_text('⏸ Pause')
            self.btn_play.color = '#f38ba8'
            self.btn_play.hovercolor = '#eba0ac'
            self.start_timer()
        else:
            self.btn_play.label.set_text('▶ Play')
            self.btn_play.color = '#a6e3a1'
            self.btn_play.hovercolor = '#94e2d5'
            self.stop_timer()

    def start_timer(self):
        """Periodic timer trigger for play animation."""
        if self.is_playing:
            self.next_step()
            self.timer = self.fig.canvas.new_timer(interval=500)  # 500ms per step
            self.timer.add_callback(self.on_timer_tick)
            self.timer.start()

    def on_timer_tick(self):
        """Timer tick callback."""
        if self.is_playing:
            max_step = len(self.current_exp_traj) - 1 if self.current_exp_traj is not None else 0
            if self.current_step >= max_step:
                self.current_step = 0  # Loop playback
            else:
                self.current_step += 1
            self.slider.set_val(self.current_step)

    def stop_timer(self):
        if self.timer:
            self.timer.stop()
            self.timer = None

    def on_exp_submit(self, text):
        """Handle switching to a new experiment ID via text box."""
        try:
            new_id = int(text)
            self.load_experiment(new_id)
            num_steps = len(self.current_exp_traj) if not self.current_exp_traj.empty else 1
            self.slider.valmax = max(0, num_steps - 1)
            self.slider.set_val(0)
            self.render_plot()
        except ValueError:
            print(f"Invalid experiment ID: {text}")


def main():
    parser = argparse.ArgumentParser(description="UAV & Sensor Experiment Visualizer")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"c:\Users\user\Documents\BGU-Master\Transformer-based-UAV-Localization-with-Multi-Sensors\Data\test_data_10_sensors_zsectors_ptx_model_preprocessing\5K_4",
        help="Path to directory containing experiment CSV files"
    )
    parser.add_argument("--exp_id", type=int, default=1, help="Initial Experiment ID to visualize")
    args = parser.parse_args()

    visualizer = ExperimentVisualizer(data_dir=args.data_dir, exp_id=args.exp_id)
    plt.show()


if __name__ == "__main__":
    main()
