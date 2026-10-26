# ============================================================
# ARGO 3D EAM MAP
# ============================================================
# Creates a 3D EAM-style mesh from the ARGO MATLAB dataset.
# The mesh is coloured by Voltage, with optional LAT figure.
# ============================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.io as sio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# AUTOMATICALLY FIND ARGODataset_MATLAB.mat
# ============================================================

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("=" * 70)
print("SEARCHING FOR ARGO MATLAB FILE")
print("=" * 70)
print("Project root:", PROJECT_ROOT)

# Search the complete project for the MATLAB file
mat_candidates = list(
    PROJECT_ROOT.rglob("ARGODataset_MATLAB.mat")
)

# If exact filename is not found, search for any .mat file
if not mat_candidates:
    print("\nExact ARGODataset_MATLAB.mat not found.")
    print("Searching for all MATLAB files...")

    mat_candidates = list(
        PROJECT_ROOT.rglob("*.mat")
    )

if not mat_candidates:
    print("\nERROR: No .mat file was found anywhere inside the project.")
    print("\nPlease check your ARGO dataset extraction.")
    sys.exit(1)

print("\nMATLAB files found:")

for i, p in enumerate(mat_candidates, start=1):
    print(f"{i}. {p}")

# ------------------------------------------------------------
# Prefer ARGODataset_MATLAB.mat
# ------------------------------------------------------------

exact_matches = [
    p for p in mat_candidates
    if p.name.lower() == "argodataset_matlab.mat"
]

if exact_matches:
    MAT_PATH = exact_matches[0]
else:
    # Use the first .mat file if exact name differs
    MAT_PATH = mat_candidates[0]

print("\nUsing MATLAB file:")
print(MAT_PATH)
print("=" * 70)

RESULT_DIR = PROJECT_ROOT / "results" / "ARGO_ALL_RESULTS" / "EAM_3D"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Choose patient 1 for the requested EAM diagram.
PATIENT_INDEX = 0

mat = sio.loadmat(
    MAT_PATH,
    squeeze_me=True,
    struct_as_record=False
)

dataset = mat["ARGODataset_MATLAB"]
patient = np.atleast_1d(dataset)[PATIENT_INDEX]

xyz = patient.XYZmesh
conn = patient.ConnectivityList
coloring = patient.MESHcoloring
ablation = patient.AblationPoints

x = np.asarray(xyz.X, dtype=float).ravel()
y = np.asarray(xyz.Y, dtype=float).ravel()
z = np.asarray(xyz.Z, dtype=float).ravel()

tri = np.column_stack(
    [
        np.asarray(conn.node1).ravel(),
        np.asarray(conn.node2).ravel(),
        np.asarray(conn.node3).ravel(),
    ]
).astype(int) - 1

voltage = np.asarray(
    coloring.Voltage,
    dtype=float
).ravel()

lat = np.asarray(
    coloring.LAT,
    dtype=float
).ravel()

# Remove invalid triangles.
valid_vertex = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
valid_tri = np.all(valid_vertex[tri], axis=1)

tri = tri[valid_tri]

# Build triangular faces.
vertices = np.column_stack([x, y, z])
faces = vertices[tri]

# ------------------------------------------------------------
# Voltage EAM
# ------------------------------------------------------------

face_voltage = np.nanmean(
    voltage[tri],
    axis=1
)

valid_face_voltage = np.isfinite(face_voltage)
faces_voltage = faces[valid_face_voltage]
face_voltage = face_voltage[valid_face_voltage]

norm = colors.Normalize(
    vmin=np.nanpercentile(face_voltage, 2),
    vmax=np.nanpercentile(face_voltage, 98)
)

face_colors = cm.viridis(norm(face_voltage))

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

mesh = Poly3DCollection(
    faces_voltage,
    facecolors=face_colors,
    linewidths=0.05,
    alpha=0.90
)

ax.add_collection3d(mesh)

# Add ablation points if available.
ax.scatter(
    np.asarray(ablation.X).ravel(),
    np.asarray(ablation.Y).ravel(),
    np.asarray(ablation.Z).ravel(),
    s=10,
    c="red",
    alpha=0.85,
    label="Ablation Points"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(
    "ARGO 3D EAM Voltage Map - Patient 1",
    fontsize=16,
    fontweight="bold"
)

ax.set_box_aspect(
    (
        np.ptp(x),
        np.ptp(y),
        np.ptp(z)
    )
)

mappable = cm.ScalarMappable(
    norm=norm,
    cmap="viridis"
)
mappable.set_array(face_voltage)
fig.colorbar(
    mappable,
    ax=ax,
    shrink=0.65,
    pad=0.08,
    label="Voltage"
)

ax.legend()
plt.tight_layout()

voltage_path = RESULT_DIR / "ARGO_Pt1_3D_EAM_Voltage.png"
plt.savefig(
    voltage_path,
    dpi=400,
    bbox_inches="tight"
)
plt.show()

# ------------------------------------------------------------
# LAT EAM
# ------------------------------------------------------------

face_lat = np.nanmean(
    lat[tri],
    axis=1
)

valid_face_lat = np.isfinite(face_lat)
faces_lat = faces[valid_face_lat]
face_lat = face_lat[valid_face_lat]

norm_lat = colors.Normalize(
    vmin=np.nanpercentile(face_lat, 2),
    vmax=np.nanpercentile(face_lat, 98)
)

face_colors_lat = cm.plasma(norm_lat(face_lat))

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

mesh = Poly3DCollection(
    faces_lat,
    facecolors=face_colors_lat,
    linewidths=0.05,
    alpha=0.90
)

ax.add_collection3d(mesh)

ax.scatter(
    np.asarray(ablation.X).ravel(),
    np.asarray(ablation.Y).ravel(),
    np.asarray(ablation.Z).ravel(),
    s=10,
    c="red",
    alpha=0.85,
    label="Ablation Points"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(
    "ARGO 3D EAM Local Activation Time Map - Patient 1",
    fontsize=16,
    fontweight="bold"
)

ax.set_box_aspect(
    (
        np.ptp(x),
        np.ptp(y),
        np.ptp(z)
    )
)

mappable = cm.ScalarMappable(
    norm=norm_lat,
    cmap="plasma"
)
mappable.set_array(face_lat)
fig.colorbar(
    mappable,
    ax=ax,
    shrink=0.65,
    pad=0.08,
    label="LAT"
)

ax.legend()
plt.tight_layout()

lat_path = RESULT_DIR / "ARGO_Pt1_3D_EAM_LAT.png"
plt.savefig(
    lat_path,
    dpi=400,
    bbox_inches="tight"
)
plt.show()

print("Saved:")
print(voltage_path)
print(lat_path)
