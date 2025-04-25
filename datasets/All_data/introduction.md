# Liver Mesh Landmark Dataset and Annotation Tutorial

This directory provides a carefully curated dataset of **200 liver surface meshes** annotated with two anatomical landmarks: the **falciform ligament** and the **liver ridge**. The dataset was constructed from public CT datasets and is accompanied by annotation scripts and documentation, enabling reproducible 3D mesh labeling for geometric deep learning research in surgical navigation.

---

## 📁 Directory Structure

All_data/ ├── 3DMeshAnnotationTutorial/ # Annotation scripts and step-by-step tutorial │ ├── LabelLandmarks-Blender.py # Blender script to annotate liver mesh vertices/edges │ └── annotating-edges-on-a-3D-liver-mesh.pdf # Full annotation workflow guide ├── livermesh/ # 200 liver mesh files (.obj), reconstructed from CT ├── edges/ # Each mesh's edge list (vertex pairs) ├── seg/ # Hard segmentation labels: 1=background, 2=ligament, 3=ridge ├── sseg/ # Soft segmentation labels (for compatibility; not used in this study) ├── classes.txt # Label definition file └── mean_std_cache.p # Feature normalization statistics (used by MeshGraphCNN)


---

## 📌 Dataset Description

- **Source:** Meshes were reconstructed from three public CT datasets: 3Dircadb, MSD Task08 (LiTS), and AMOS22.
- **Landmarks:** Two key anatomical landmarks—**falciform ligament** and **liver ridge**—are annotated as edge-level labels on surface meshes.
- **Label Format:** 
  - `seg/` contains integer labels for each edge.
  - `sseg/` contains soft label vectors (not used in our current model).
- **Mesh Format:** Liver surfaces are saved in `.obj` format, each accompanied by an edge list and annotation.

---

## 🛠️ Annotation Workflow

A full tutorial is provided in [`3DMeshAnnotationTutorial/annotating-edges-on-a-3D-liver-mesh.pdf`](3DMeshAnnotationTutorial/annotating-edges-on-a-3D-liver-mesh.pdf), which explains:

- How to extract liver surface meshes from CT scans
- How to manually annotate liver landmarks in **3D Slicer** and **Blender**
- How to use the provided Blender script [`LabelLandmarks-Blender.py`](3DMeshAnnotationTutorial/LabelLandmarks-Blender.py) to generate edge-level labels

---

## 🧪 Suggested Use

This dataset supports the training and evaluation of anatomical landmark segmentation algorithms on liver meshes, particularly under geometric deep learning frameworks such as **MeshCNN** and **MeshGraphCNN**.

---

If you use this dataset or annotation protocol in your research, please consider citing the corresponding publication.


