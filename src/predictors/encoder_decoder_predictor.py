"""
encoder_decoder_predictor.py

Implementation of the Encoder-Decoder UAV Localization Predictor.
Processes raw experiment dataset input features directly and enforces all feature
normalization, scaling, and sequence densification internally.
"""

import os
import sys
import ast
import re
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .base_predictor import BaseUAVPredictor
from ..models.hybrid_encoder import HybridSpatiotemporalPredictor
from ..models.lstm_decoder import LSTMDecoder


class EncoderDecoderPredictor(BaseUAVPredictor):
    """
    Encoder-Decoder UAV Localization Predictor.
    Combines Transformer Encoder and LSTM Decoder.
    Processes raw experiment dataset input features directly.
    """

    def __init__(
        self,
        enc_ckpt_path: Optional[str] = None,
        dec_ckpt_path: Optional[str] = None,
        device: Optional[str] = None,
        window_size: int = 40,
        num_sensors: int = 10,
        name: str = "Encoder-Decoder (Epoch 16/12)",
    ):
        super().__init__(name=name)
        self.window_size = window_size
        self.num_sensors = num_sensors

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

        enc_filename = "hybrid_predictor_10sensors_16_epochs.pth"
        dec_filename = "decoder_10sensors_12_epochs.pth"

        search_dirs_enc = [
            os.path.join(root_dir, "models", "encoder"),
            os.path.join(root_dir, "models"),
            os.path.join(root_dir, "Encoder-Decoder", "Encoder - Transformer"),
            os.path.join(root_dir, "Encoder-Decoder"),
        ]

        search_dirs_dec = [
            os.path.join(root_dir, "models", "decoder"),
            os.path.join(root_dir, "models"),
            os.path.join(root_dir, "Encoder-Decoder", "Decoder - LSTM"),
            os.path.join(root_dir, "Encoder-Decoder"),
        ]

        self.enc_ckpt_path = self._resolve_checkpoint(
            specified_path=enc_ckpt_path,
            target_filename=enc_filename,
            search_dirs=search_dirs_enc,
            description="Transformer Encoder checkpoint (Epoch 16)"
        )

        self.dec_ckpt_path = self._resolve_checkpoint(
            specified_path=dec_ckpt_path,
            target_filename=dec_filename,
            search_dirs=search_dirs_dec,
            description="LSTM Decoder checkpoint (Epoch 12)"
        )

        self.encoder, self.decoder = self._load_models()

    @staticmethod
    def _resolve_checkpoint(
        specified_path: Optional[str],
        target_filename: str,
        search_dirs: list,
        description: str
    ) -> str:
        if specified_path:
            if os.path.exists(specified_path):
                return os.path.abspath(specified_path)
            else:
                raise FileNotFoundError(f"{description} specified at '{specified_path}' does not exist.")

        for d in search_dirs:
            candidate = os.path.join(d, target_filename)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        checked_str = "\n".join(f" - {os.path.join(d, target_filename)}" for d in search_dirs)
        raise FileNotFoundError(
            f"{description} '{target_filename}' not found.\n"
            f"Checked locations:\n{checked_str}"
        )

    def _load_models(self) -> Tuple[nn.Module, nn.Module]:
        encoder = HybridSpatiotemporalPredictor(
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            dynamic_dim=6,
            static_dim=3,
            coords_dim=3,
        )
        if not os.path.exists(self.enc_ckpt_path):
            raise FileNotFoundError(f"Encoder checkpoint not found at: {self.enc_ckpt_path}")

        encoder.load_state_dict(torch.load(self.enc_ckpt_path, map_location=self.device))
        encoder.to(self.device)
        encoder.eval()

        decoder = LSTMDecoder(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.1,
            output_size=3,
        )
        if not os.path.exists(self.dec_ckpt_path):
            raise FileNotFoundError(f"Decoder checkpoint not found at: {self.dec_ckpt_path}")

        dec_ckpt = torch.load(self.dec_ckpt_path, map_location=self.device)
        state_dict = dec_ckpt.get("decoder_state_dict", dec_ckpt)
        decoder.load_state_dict(state_dict)
        decoder.to(self.device)
        decoder.eval()

        return encoder, decoder

    @staticmethod
    def _parse_sector(val) -> int:
        if pd.isna(val):
            return 1
        if isinstance(val, (int, float, np.integer, np.floating)):
            return int(val)
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return int(parsed[0])
            except Exception:
                pass
            match = re.search(r"\d+", val)
            if match:
                return int(match.group(0))
        return 1

    def preprocess_raw_data(self, raw_exp_data: Dict[str, Any]) -> pd.DataFrame:
        det_df = raw_exp_data["detections"].copy()
        info_df = raw_exp_data.get("sensor_info", pd.DataFrame())
        dist_df = raw_exp_data.get("sensor_dist", pd.DataFrame())

        det_exp_col = "experiment_id" if "experiment_id" in det_df.columns else "exp_id"
        exp_id = det_df[det_exp_col].iloc[0] if not det_df.empty else 1

        exp_det = det_df[det_df[det_exp_col] == exp_id].copy()

        dist_exp_col = "exp_id" if "exp_id" in dist_df.columns else ("experiment_id" if "experiment_id" in dist_df.columns else (dist_df.columns[0] if not dist_df.empty else None))
        exp_dist = dist_df[dist_df[dist_exp_col] == exp_id] if dist_exp_col and not dist_df.empty else pd.DataFrame()

        pos_map = {}
        if not exp_dist.empty:
            for _, r in exp_dist.iterrows():
                s1, s2 = int(r["sensor1_id"]), int(r["sensor2_id"])
                if s1 not in pos_map and "x1" in r:
                    pos_map[s1] = (float(r["x1"]) / 1000.0, float(r["y1"]) / 1000.0, float(r["z1"]) / 1000.0)
                if s2 not in pos_map and "x2" in r:
                    pos_map[s2] = (float(r["x2"]) / 1000.0, float(r["y2"]) / 1000.0, float(r["z2"]) / 1000.0)

        static_map = {}
        if not info_df.empty:
            for _, r in info_df.iterrows():
                s_id = int(r["sensor_id"])
                gain_val = float(r["gain"])
                sinr_val = float(r["sinr_threshold"]) / 10.0
                sens_val = float(r["sensitivity"]) / 90.0
                static_map[s_id] = (gain_val, sinr_val, sens_val)

        unique_ts = np.sort(exp_det["timestamp"].unique())
        ts_map = {t: idx for idx, t in enumerate(unique_ts)}

        sec_col = "sensor_sectors_detected" if "sensor_sectors_detected" in exp_det.columns else "sector_detected"
        pow_col = "total_power_dBm" if "total_power_dBm" in exp_det.columns else "received_power"

        exp_det["time_step"] = exp_det["timestamp"].map(ts_map)
        exp_det["sector"] = exp_det[sec_col].apply(self._parse_sector)

        exp_sensors = np.sort(exp_det["sensor_id"].unique())
        if len(exp_sensors) != self.num_sensors:
            if not exp_dist.empty:
                s1_ids = set(exp_dist["sensor1_id"].unique())
                s2_ids = set(exp_dist["sensor2_id"].unique())
                dist_sensors = np.sort(list(s1_ids.union(s2_ids)))
                if len(dist_sensors) == self.num_sensors:
                    exp_sensors = dist_sensors
                else:
                    base_sid = ((exp_id - 1) * self.num_sensors) + 1
                    exp_sensors = np.arange(base_sid, base_sid + self.num_sensors, dtype=int)
            else:
                base_sid = ((exp_id - 1) * self.num_sensors) + 1
                exp_sensors = np.arange(base_sid, base_sid + self.num_sensors, dtype=int)

        skeleton = pd.MultiIndex.from_product(
            [[exp_id], range(len(unique_ts)), exp_sensors],
            names=[det_exp_col, "time_step", "sensor_id"]
        ).to_frame().reset_index(drop=True)

        pivot = exp_det.pivot_table(
            index=[det_exp_col, "time_step", "sensor_id"],
            columns="sector",
            values=pow_col,
            fill_value=0.0
        ).reset_index()

        for i in range(1, 7):
            if i not in pivot.columns:
                pivot[i] = 0.0

        sector_cols = [f"sector_{i}_power" for i in range(1, 7)]
        pivot = pivot.rename(columns={i: f"sector_{i}_power" for i in range(1, 7)})

        merged = pd.merge(skeleton, pivot, on=[det_exp_col, "time_step", "sensor_id"], how="left")
        merged[sector_cols] = merged[sector_cols].fillna(0.0)

        merged["sensor_x"] = merged["sensor_id"].map(lambda s: pos_map.get(s, (0.0, 0.0, 0.0))[0])
        merged["sensor_y"] = merged["sensor_id"].map(lambda s: pos_map.get(s, (0.0, 0.0, 0.0))[1])
        merged["sensor_z"] = merged["sensor_id"].map(lambda s: pos_map.get(s, (0.0, 0.0, 0.0))[2])

        merged["gain"] = merged["sensor_id"].map(lambda s: static_map.get(s, (0.0, 0.0, 0.0))[0])
        merged["sinr_threshold"] = merged["sensor_id"].map(lambda s: static_map.get(s, (0.0, 0.0, 0.0))[1])
        merged["sensitivity"] = merged["sensor_id"].map(lambda s: static_map.get(s, (0.0, 0.0, 0.0))[2])

        cols = [
            det_exp_col, "sensor_id", "time_step",
            "sector_1_power", "sector_2_power", "sector_3_power",
            "sector_4_power", "sector_5_power", "sector_6_power",
            "sensor_x", "sensor_y", "sensor_z",
            "gain", "sinr_threshold", "sensitivity"
        ]

        res = merged[cols].sort_values(by=[det_exp_col, "time_step", "sensor_id"]).reset_index(drop=True)
        return res

    def predict_experiment(self, raw_exp_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
        norm_df = self.preprocess_raw_data(raw_exp_data)
        exp_col = norm_df.columns[0]
        exp_id = norm_df[exp_col].iloc[0] if not norm_df.empty else 1

        exp_df = norm_df[norm_df[exp_col] == exp_id].sort_values(by=["time_step", "sensor_id"]).reset_index(drop=True)
        feat_vals = exp_df.values.astype(np.float32)

        start_time = time.perf_counter()

        x_tensor = torch.tensor(feat_vals).unsqueeze(0)
        total_rows = x_tensor.size(1)

        if total_rows < self.window_size:
            pad_len = self.window_size - total_rows
            pad_tensor = torch.zeros(1, pad_len, 15, dtype=x_tensor.dtype)
            x_tensor = torch.cat([pad_tensor, x_tensor], dim=1)
            total_rows = self.window_size

        windows = [x_tensor[:, i : i + self.window_size] for i in range(total_rows - self.window_size + 1)]
        windows_cat = torch.cat(windows, dim=0).to(self.device)

        dynamic_features = windows_cat[:, :, 3:9]
        spatial_coords   = windows_cat[:, :, 9:12]
        static_features  = windows_cat[:, :, 12:15]

        ts_arr = windows_cat[:, -1, 2].cpu().numpy().astype(int)

        batch_size = 512
        all_preds = []

        with torch.no_grad():
            for b_idx in range(0, len(windows_cat), batch_size):
                b_dyn = dynamic_features[b_idx : b_idx + batch_size]
                b_stat = static_features[b_idx : b_idx + batch_size]
                b_spat = spatial_coords[b_idx : b_idx + batch_size]

                enc_hidden = self.encoder.get_hidden_state(b_dyn, b_stat, b_spat, layer_idx="embedding")
                pred_seq = self.decoder(enc_hidden)
                pred_last = pred_seq[:, -1, :].cpu().numpy()
                all_preds.append(pred_last)

        raw_preds_km = np.concatenate(all_preds, axis=0)
        processing_time_sec = time.perf_counter() - start_time

        raw_preds_m = raw_preds_km * 1000.0

        pred_df = pd.DataFrame({
            "timestamp": ts_arr,
            "pred_x": raw_preds_m[:, 0],
            "pred_y": raw_preds_m[:, 1],
            "pred_z": raw_preds_m[:, 2],
        })

        agg_df = pred_df.groupby("timestamp").mean().reset_index()

        final_pred_m = agg_df[["pred_x", "pred_y", "pred_z"]].values
        final_timestamps = agg_df["timestamp"].values

        return final_pred_m, final_timestamps, processing_time_sec
