DIP_PRO/
│
├── data/                     
│   └── (training/testing images)
│
├── results/                  
│   └── (model outputs, predictions)
│
├── src/
│   ├── baselines.py          → Classical pseudo-colorization (LUT, slicing, etc.)
│   ├── gui.py                → GUI app for pseudo-colorization
│   ├── model.py              → Deep Learning model (U-Net)
│   └── run_all.py            → Full pipeline (load → predict → display)
│
├── requirements.txt          → Dependencies
│
├── dip-project(model train notebook).ipynb   
│       → Jupyter notebook used for model training
│
└── Documentation.docx        → Full project documentation





🚀 Features

✔ Deep Learning pseudo colorization using modified U-Net

✔ GUI application for easy image colorization

✔ Baseline pseudo-colorization (LUT, level slicing)

✔ Training notebook included

✔ Triang Model trained on 7000+ images datset with 37 + clases 

✔ Outputs saved automatically in results/

✔ Full documentation included


🧠 Model Overview

This project uses a U-Net Convolutional Neural Network:

Input: 1-channel grayscale image

Output: 3-channel pseudo-colored image

Framework: Pytorch

**🛠 Installation**


git clone https://github.com/<your-username>/pseudo-colorization-DIP.git

after clone the repo  cd DIP_PRO

Install the requiremnts.txt

python .\src\run_all.py --gui  (for run the project )
