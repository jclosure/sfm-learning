from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from sfm_learning.pipeline import run_pipeline


st.set_page_config(page_title="SfM Learning", layout="wide")
st.title("SfM Learning Demo")
st.caption("Upload a sequence of overlapping photos to run sparse incremental reconstruction.")

files = st.file_uploader("Images", accept_multiple_files=True, type=["jpg", "jpeg", "png", "webp"])
run = st.button("Run reconstruction")

if run and files:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for f in files:
            (d / f.name).write_bytes(f.read())

        with st.spinner("Running pipeline..."):
            result = run_pipeline(d)

        pts = np.array([p.xyz for p in result.reconstruction.points.values()])
        if len(pts) == 0:
            st.warning("No points reconstructed. Try more overlap and texture.")
        else:
            fig = go.Figure(data=[go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers", marker=dict(size=2)
            )])
            fig.update_layout(height=700, scene_aspectmode="data")
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Poses: {len(result.reconstruction.poses)} | Points: {len(result.reconstruction.points)}")
