# Nested Resolution Mesh-Graph CNN for Liver Landmark Segmentation

This repository provides the implementation and dataset used in our study:

> **Nested Resolution Mesh-Graph CNN for Automated Extraction of Liver Surface Anatomical Landmarks**  

We describe a mesh-based deep learning framework for automatically segmenting anatomical landmarks—specifically the **falciform ligament** and **liver ridge**—on 3D liver meshes. The model combines global geometric learning and local anatomical refinement using dynamic graph convolution (DGCNN) and mesh convolution (MeshConv), aiming to support downstream applications such as **AR-assisted surgical navigation**.

---

## 🔍 Highlights

- ⚙️ A novel nested resolution Mesh-Graph CNN is proposed for the segmentation of liver surface anatomical landmarks.
- 🧠 Seamlessly integrated global shape analysis with local topological refinement is proposed to enhance segmentation accuracy.
- 🏷️ An attention fusion module with auxiliary supervision adaptively combines multi-threshold landmark proposals, enhancing spatial consistency and anatomical plausibility.
- 📊 200 liver meshes are annotated that are used to both develop and validate our methods.
- ⚙️ Experiments demonstrate superior performance of our method over state-of-the-art in both internal and external datasets.

---

## 📁 Repository Structure

```bash
MeshGraphCNN/
├── datasets/
│   └── All_data/
│       ├── livermesh/                # Liver mesh files (.obj)
│       ├── seg/                      # Edge-wise landmark labels: 1=background, 2=ligament, 3=ridge
│       ├── sseg/                     # Soft labels (optional)
│       ├── edges/                    # Mesh edge list per file
│       ├── classes.txt               # Label definitions
│       ├── mean_std_cache.p          # MeshCNN normalization cache
│       └── 3DMeshAnnotationTutorial/
│           ├── LabelLandmarks-Blender.py
│           └── annotating-edges-on-a-3D-liver-mesh.pdf
├── PyTorch3D-3D-2D-Registration/
│   ├── run_p2ilf_7.py                # PyTorch3D-based 3D-2D registration demo
│   ├── obj/                          # Liver mesh for registration
│   ├── camera-parameter/            # Camera intrinsics
│   ├── image-2d-landmark/           # 2D laparoscopic landmark inputs
│   └── RegistrationFramework.png    # Framework overview figure
├── train.py
├── test.py
└── ...
```
